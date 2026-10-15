using System;
using System.Collections;
using System.Collections.Generic;
using System.Reflection;
using GaussianSplatting.Editor.Utils;
using NUnit.Framework;
using UnityEditor;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.TestTools;

namespace GaussianSplatting.Tests
{
    public class ColorPrecisionTests
    {
        string folder;
        readonly List<Texture2D> transient = new List<Texture2D>();
        static readonly Color Hdr = new Color(1.5f, -.25f, .33325f, .6f);
        static readonly Color Sh = new Color(.314159f, .607123f, .901234f, 0);

        [SetUp]
        public void SetUp()
        {
            string name = "__GSColorTest_" + Guid.NewGuid().ToString("N");
            AssetDatabase.CreateFolder("Assets", name);
            folder = "Assets/" + name;
        }

        [TearDown]
        public void TearDown()
        {
            foreach (var texture in transient) if (texture != null) UnityEngine.Object.DestroyImmediate(texture);
            transient.Clear();
            AssetDatabase.DeleteAsset(folder);
        }

        Texture2D Source(TextureFormat format, Color value)
        {
            bool compressed = format == TextureFormat.BC7 || format == TextureFormat.DXT1;
            var texture = new Texture2D(4, 4, compressed ? TextureFormat.RGBA32 : format, false, true);
            var pixels = new Color[16];
            for (int i = 0; i < pixels.Length; i++) pixels[i] = value;
            texture.SetPixels(pixels); texture.Apply();
            if (compressed)
            {
                EditorUtility.CompressTexture(texture, format, TextureCompressionQuality.Normal);
                texture.Apply(false, false);
            }
            transient.Add(texture);
            return texture;
        }

        static Color Pixel(Texture2D source, int x = 0)
        {
            var rt = RenderTexture.GetTemporary(source.width, source.height, 0, RenderTextureFormat.ARGBFloat, RenderTextureReadWrite.Linear);
            var active = RenderTexture.active;
            var read = new Texture2D(source.width, source.height, TextureFormat.RGBAFloat, false, true);
            try
            {
                Graphics.Blit(source, rt); RenderTexture.active = rt;
                read.ReadPixels(new Rect(0, 0, source.width, source.height), 0, 0); read.Apply();
                return read.GetPixel(x, 0);
            }
            finally
            {
                RenderTexture.active = active; RenderTexture.ReleaseTemporary(rt);
                UnityEngine.Object.DestroyImmediate(read);
            }
        }

        static void AssertColor(Color actual, Color expected, float tolerance)
        {
            Assert.That(actual.r, Is.EqualTo(expected.r).Within(tolerance));
            Assert.That(actual.g, Is.EqualTo(expected.g).Within(tolerance));
            Assert.That(actual.b, Is.EqualTo(expected.b).Within(tolerance));
        }

        [TestCase(false, TextureFormat.RGBAHalf)]
        [TestCase(true, TextureFormat.RGBA32)]
        public void ColorFormat_HonorsCompressionChoice(bool compress, TextureFormat expected)
            => Assert.That(GaussianSplatImporter.ColorSourceTextureFormat(compress), Is.EqualTo(expected));

        [TestCase(GaussianSplatImporter.SHCompression.None, TextureFormat.RGBAHalf)]
        [TestCase(GaussianSplatImporter.SHCompression.BC1, TextureFormat.RGBA32)]
        [TestCase(GaussianSplatImporter.SHCompression.BC7, TextureFormat.RGBA32)]
        public void ShFormat_HonorsCompressionChoice(GaussianSplatImporter.SHCompression compression, TextureFormat expected)
            => Assert.That(GaussianSplatImporter.SHSourceTextureFormat(compression), Is.EqualTo(expected));

        object Build(bool compressColor, GaussianSplatImporter.SHCompression compression)
        {
            var type = Assembly.Load("GaussianSplatting.Editor").GetType("GaussianSplatting.GaussianSplatLODImporter+StreamedTextureSetBuilder", true);
            object builder = Activator.CreateInstance(type, new object[] { folder, "precision", 0, 16, true, 3, Vector4.zero, Vector4.one, compression, compressColor });
            var splat = new ImportSplatData { pos = Vector3.zero, rot = Quaternion.identity, scale = Vector3.one, dc0 = new Vector3(Hdr.r, Hdr.g, Hdr.b), opacity = 1.4f };
            float[] sh = { Sh.r, Sh.g, Sh.b, Sh.r, Sh.g, Sh.b, Sh.r, Sh.g, Sh.b };
            type.GetMethod("WriteSplat").Invoke(builder, new object[] { 0, splat, Vector3.zero, Vector3.one, sh, 0L });
            type.GetMethod("Save").Invoke(builder, null);
            return builder;
        }

        static Texture2D Texture(object builder, string property)
            => (Texture2D)builder.GetType().GetProperty(property).GetValue(builder);

        [Test]
        public void Lod_UncompressedRetainsHdrDcAndShPrecision()
        {
            object builder = Build(false, GaussianSplatImporter.SHCompression.None);
            var color = Texture(builder, "ColorTexture"); var sh = Texture(builder, "SHTexture");
            Assert.That(color.format, Is.EqualTo(TextureFormat.RGBAHalf));
            Assert.That(sh.format, Is.EqualTo(TextureFormat.RGBAHalf));
            AssertColor(Pixel(color), Hdr, .001f);
            Assert.That(Pixel(color).a, Is.EqualTo(1f)); // only opacity is clamped
            AssertColor(Pixel(sh), Sh, .0005f);
        }

        [TestCase(GaussianSplatImporter.SHCompression.BC1, TextureFormat.DXT1)]
        [TestCase(GaussianSplatImporter.SHCompression.BC7, TextureFormat.BC7)]
        public void Lod_ExplicitCompressionKeepsCompactFormats(GaussianSplatImporter.SHCompression mode, TextureFormat expected)
        {
            object builder = Build(true, mode);
            Assert.That(Texture(builder, "ColorTexture").format, Is.EqualTo(TextureFormat.BC7));
            Assert.That(Texture(builder, "SHTexture").format, Is.EqualTo(expected));
        }

        IEnumerator Concat(string attribute, bool firstHalf, bool secondHalf, TextureFormat expected, Color value)
        {
            Assert.That(SystemInfo.supportsAsyncGPUReadback, Is.True, "GPU async readback is required for this regression test");
            var fuse = typeof(GaussianSplatFuse);
            var regionType = fuse.GetNestedType("FileRegion", BindingFlags.NonPublic);
            var attrType = fuse.GetNestedType("FileAttr", BindingFlags.NonPublic);
            var regions = (IList)Activator.CreateInstance(typeof(List<>).MakeGenericType(regionType));
            for (int i = 0; i < 2; i++)
            {
                bool half = i == 0 ? firstHalf : secondHalf;
                TextureFormat format = half ? TextureFormat.RGBAHalf : attribute == "SH" ? TextureFormat.BC7 : TextureFormat.RGBA32;
                object region = Activator.CreateInstance(regionType);
                regionType.GetField("fusedBase").SetValue(region, i);
                regionType.GetField("count").SetValue(region, 1);
                regionType.GetField(attribute == "SH" ? "pos" : "col").SetValue(region, Source(format, value));
                regions.Add(region);
            }
            var type = fuse.GetNestedType("AsyncConcatReadback", BindingFlags.NonPublic);
            object job = type.GetMethod("Start").Invoke(null, new object[] { 4, 4, regions, Enum.Parse(attrType, attribute), "concat", folder });
            var step = type.GetMethod("Step");
            var args = new object[] { null };
            Texture2D result = null;
            try
            {
                double deadline = EditorApplication.timeSinceStartup + 30;
                while (!(bool)step.Invoke(job, args))
                {
                    Assert.That(EditorApplication.timeSinceStartup, Is.LessThan(deadline), "GPU readback timeout");
                    yield return null;
                }
                result = (Texture2D)args[0];
                Assert.That(result.format, Is.EqualTo(expected));
                if (firstHalf) AssertColor(Pixel(result), value, .001f);
                else if (secondHalf)
                {
                    // A later half-float source must promote the output even when the first source is compact.
                    AssertColor(Pixel(result, 1), value, .001f);
                }
            }
            finally { if (result == null) type.GetMethod("Dispose").Invoke(job, null); }
        }

        [UnityTest] public IEnumerator Fuse_UncompressedColorRetainsHdr() => Concat("Col", true, true, TextureFormat.RGBAHalf, Hdr);
        [UnityTest] public IEnumerator Fuse_UncompressedShRetainsPrecision() => Concat("SH", true, true, TextureFormat.RGBAHalf, Sh);
        [UnityTest] public IEnumerator Fuse_CompactColorStaysRgba32() => Concat("Col", false, false, TextureFormat.RGBA32, Sh);
        [UnityTest] public IEnumerator Fuse_CompressedShStaysBc7() => Concat("SH", false, false, TextureFormat.BC7, Sh);
        [UnityTest] public IEnumerator Fuse_MixedColorPromotesOutput() => Concat("Col", false, true, TextureFormat.RGBAHalf, Hdr);
        [UnityTest] public IEnumerator Fuse_MixedShPromotesOutput() => Concat("SH", false, true, TextureFormat.RGBAHalf, Sh);
    }
}
