using System.Collections.Generic;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using OpenCvSharp;

namespace ConsciousCar
{
    public class RoadDetector
    {
        private class OUTPUT
        {
            public const int Width = 600;
            public const int Height = 600;

            [ColumnName("masks")]
            public float[] Values { get; set; }
        }

        private class INPUT
        {
            public const int ChannelAmount = 3;
            public const int Width = 600;
            public const int Height = 600;

            [ColumnName("input")]
            [VectorType(1, ChannelAmount, Width, Height)]
            public float[] Values { get; set; }
        }

        private readonly PredictionEngine<INPUT, OUTPUT> predictionEngine;

        public RoadDetector()
        {
            var mlContext = new MLContext();

            var gpuModelEstimator = mlContext.Transforms.ApplyOnnxModel(outputColumnName: "masks", inputColumnName: "input", modelFile: "model.onnx", gpuDeviceId: 0);

            var data = mlContext.Data.LoadFromEnumerable(new List<INPUT>());

            var gpuModel = gpuModelEstimator.Fit(data);

            predictionEngine = mlContext.Model.CreatePredictionEngine<INPUT, OUTPUT>(gpuModel, ignoreMissingColumns: false);
        }

        private static float[] ProcessImage(Vec3b[] rgbImage)
        {
            var pixcels = new List<float>();
            foreach (var pixcel in rgbImage)
            {
                pixcels.Add(((float)pixcel.Item0) / 255 );
            }
            foreach (var pixcel in rgbImage)
            {
                pixcels.AddRange(new[] { ((float)pixcel.Item1) / 255 });
            }
            foreach (var pixcel in rgbImage)
            {
                pixcels.AddRange(new[] { ((float)pixcel.Item2) / 255 });
            }

            return pixcels.ToArray();
        }

        public IEnumerable<float[]> DetectNext(Mat image)
        {
            Cv2.Resize(image.Clone(), image, new Size(INPUT.Width, INPUT.Height));            

            image.GetArray<Vec3b>(out var vectorizedImage);
            var pixcelsArray = ProcessImage(vectorizedImage);

            var input = new INPUT
            {
                Values = pixcelsArray
            };

            var result = predictionEngine.Predict(input).Values;

            var resultMasks = new List<float[]>();
            for (int i = 0; i < 1; i++)
            {
                resultMasks.Add(
                    result.Skip(i * OUTPUT.Width * OUTPUT.Height)
                    .Take(OUTPUT.Width * OUTPUT.Height)
                    .ToArray()
                    );
            }

            return resultMasks.AsEnumerable();
        }
    }
}
