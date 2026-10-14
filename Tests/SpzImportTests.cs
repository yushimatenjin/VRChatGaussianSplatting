using System;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Text;
using NUnit.Framework;
using UnityEngine;

namespace GaussianSplatting.Tests
{
    public class SpzImportTests
    {
        string folder;

        [SetUp]
        public void SetUp()
        {
            folder = Path.Combine(Path.GetTempPath(), "VRCGS-SpzTests-" + Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(folder);
        }

        [TearDown]
        public void TearDown()
        {
            if (Directory.Exists(folder)) Directory.Delete(folder, true);
        }

        // Fixture arrays follow the SPZ specification's attribute-major layout.
        // All data is synthetic; no scan or external asset is required.
        string Fixture(uint version, int degree = 0, int count = 2, uint rotation = 0xC0000000u, byte flags = 0, bool truncate = false)
        {
            int coefficients = degree == 3 ? 15 : degree == 2 ? 8 : degree == 1 ? 3 : 0;
            byte[] payload;
            using (var stream = new MemoryStream())
            {
                using (var writer = new BinaryWriter(stream, Encoding.ASCII, true))
                {
                    writer.Write(0x5053474eu);
                    writer.Write(version);
                    writer.Write(count);
                    writer.Write((byte)degree);
                    writer.Write((byte)12);
                    writer.Write(flags);
                    writer.Write((byte)0);
                    for (int point = 0; point < count; point++)
                        foreach (int position in new[] { (point % 100 + 1) * 4096, -8192, 12288 })
                        {
                            writer.Write((byte)position);
                            writer.Write((byte)(position >> 8));
                            writer.Write((byte)(position >> 16));
                        }
                    for (int point = 0; point < count; point++) writer.Write((byte)(point % 2 == 0 ? 64 : 192));
                    for (int point = 0; point < count; point++) writer.Write(new byte[] { 80, 128, 200 });
                    for (int point = 0; point < count; point++) writer.Write(new byte[] { 128, 144, 160 });
                    for (int point = 0; point < count; point++)
                    {
                        if (version == 2) writer.Write(new byte[] { 128, 64, 192 });
                        else writer.Write(rotation);
                    }
                    for (int point = 0; point < count; point++)
                        for (int coefficient = 0; coefficient < coefficients; coefficient++)
                            for (int channel = 0; channel < 3; channel++)
                                writer.Write((byte)(129 + point % 2 * 48 + coefficient * 3 + channel));
                }
                payload = stream.ToArray();
            }
            string source = Path.Combine(folder, "fixture.spz");
            using (var file = File.Create(source))
            using (var gzip = new GZipStream(file, CompressionMode.Compress))
                gzip.Write(payload, 0, payload.Length - (truncate ? 1 : 0));
            return source;
        }

        float[][] Resolve(string source, int degree, int count)
        {
            string path = GaussianSplatImporter.ResolveSourceToPlyPath(source, Path.Combine(folder, "resolved"));
            byte[] data = File.ReadAllBytes(path);
            byte[] marker = Encoding.ASCII.GetBytes("end_header" + Environment.NewLine);
            int start = -1;
            for (int i = 0; i <= data.Length - marker.Length; i++)
                if (marker.Select((b, j) => data[i + j] == b).All(v => v)) { start = i + marker.Length; break; }
            Assert.That(start, Is.GreaterThan(0));
            int coefficients = degree == 3 ? 15 : degree == 2 ? 8 : degree == 1 ? 3 : 0;
            int floats = 14 + 3 * coefficients;
            Assert.That(data.Length - start, Is.EqualTo(count * floats * sizeof(float)));
            return Enumerable.Range(0, count).Select(point => Enumerable.Range(0, floats)
                .Select(i => BitConverter.ToSingle(data, start + (point * floats + i) * sizeof(float))).ToArray()).ToArray();
        }

        [TestCase(2u, 0)] [TestCase(2u, 1)] [TestCase(2u, 2)] [TestCase(2u, 3)]
        [TestCase(3u, 0)] [TestCase(3u, 1)] [TestCase(3u, 2)] [TestCase(3u, 3)]
        public void Resolve_ConvertsRubPositionsAndEveryShChannelToRdf(uint version, int degree)
        {
            var rows = Resolve(Fixture(version, degree), degree, 2);
            // Niantic RUB -> RDF signs, in the published SH coefficient order.
            int[] signs = { -1, -1, 1, -1, 1, 1, -1, 1, -1, 1, -1, -1, 1, -1, 1 };
            int coefficients = (rows[0].Length - 14) / 3;
            for (int point = 0; point < 2; point++)
            {
                Assert.That(rows[point].Take(3).ToArray(), Is.EqualTo(new float[] { point + 1, 2, -3 }));
                Assert.That(rows[point][3], Is.EqualTo((80 / 255f - .5f) / .15f).Within(1e-6));
                Assert.That(rows[point][6], Is.EqualTo(Math.Log((point == 0 ? 64f : 192f) / (point == 0 ? 191f : 63f))).Within(1e-5));
                Assert.That(rows[point].Skip(7).Take(3).ToArray(), Is.EqualTo(new float[] { -2, -1, 0 }));
                for (int channel = 0; channel < 3; channel++)
                    for (int coefficient = 0; coefficient < coefficients; coefficient++)
                    {
                        float expected = signs[coefficient] * (1 + point * 48 + coefficient * 3 + channel) / 128f;
                        Assert.That(rows[point][14 + channel * coefficients + coefficient], Is.EqualTo(expected));
                    }
            }
        }

        [TestCase(0)] [TestCase(1)] [TestCase(2)] [TestCase(3)]
        public void V3_DecodesEveryLargestQuaternionComponent(int largest)
        {
            var rows = Resolve(Fixture(3, rotation: (uint)largest << 30), 0, 2);
            float[] expected = { largest == 3 ? 1 : 0, largest == 0 ? 1 : 0, largest == 1 ? -1 : 0, largest == 2 ? -1 : 0 };
            foreach (var row in rows) Assert.That(row.Skip(10).Take(4).ToArray(), Is.EqualTo(expected));
        }

        [Test]
        public void V3_DecodesSignedMagnitudesAndConvertsRotationToRdf()
        {
            // Largest=X; Y=-181, Z=361, W=-90, all magnitudes scaled by 1/(511*sqrt(2)).
            uint packed = (693u << 20) | (361u << 10) | 602u;
            var row = Resolve(Fixture(3, rotation: packed), 0, 2)[1];
            double unit = 1 / (511 * Math.Sqrt(2));
            double x = Math.Sqrt(1 - (181 * 181 + 361 * 361 + 90 * 90) * unit * unit);
            double[] expected = { -90 * unit, x, 181 * unit, -361 * unit };
            for (int i = 0; i < 4; i++) Assert.That(row[10 + i], Is.EqualTo(expected[i]).Within(1e-6));
        }

        [Test]
        public void V2_KeepsLegacyRotationDecoderAndConvertsRotationToRdf()
        {
            var row = Resolve(Fixture(2), 0, 2)[0];
            double x = 128 / 127.5 - 1, y = 64 / 127.5 - 1, z = 192 / 127.5 - 1;
            double[] expected = { Math.Sqrt(1 - x * x - y * y - z * z), x, -y, -z };
            for (int i = 0; i < 4; i++) Assert.That(row[10 + i], Is.EqualTo(expected[i]).Within(1e-6));
        }

        [TestCase(2u)] [TestCase(3u)]
        public void Resolve_WritesAllRowsAcrossOutputBufferBoundary(uint version)
        {
            var rows = Resolve(Fixture(version, count: 8193), 0, 8193);
            Assert.That(rows[8192][0], Is.EqualTo(93));
        }

        [TestCase(2u)] [TestCase(3u)]
        public void Resolve_RejectsTruncatedAttributeStream(uint version)
        {
            string source = Fixture(version, 3, truncate: true);
            Assert.Throws<IOException>(() => GaussianSplatImporter.ResolveSourceToPlyPath(source, folder));
        }

        [TestCase(1u)] [TestCase(4u)] [TestCase(99u)]
        public void Resolve_RejectsUnsupportedVersion(uint version)
        {
            string source = Fixture(version);
            Assert.Throws<IOException>(() => GaussianSplatImporter.ResolveSourceToPlyPath(source, folder));
        }

        [TestCase(2u)] [TestCase(3u)]
        public void Resolve_AllowsAntialiasingFlag(uint version)
        {
            Assert.That(Resolve(Fixture(version, flags: 1), 0, 2).Length, Is.EqualTo(2));
        }

        [TestCase(2)] [TestCase(3)] [TestCase(128)]
        public void Resolve_RejectsUnknownFlagsInsteadOfAssumingRub(byte flags)
        {
            string source = Fixture(3, flags: flags);
            Assert.Throws<IOException>(() => GaussianSplatImporter.ResolveSourceToPlyPath(source, folder));
        }
    }
}
