using System.Collections.Generic;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using OpenCvSharp;

namespace ConsciousCar
{
    public class Detector
    {
        private class OUTPUT
        {
            public const int Width = 300;
            public const int Height = 300;

            [ColumnName("masks")]
            public float[] Masks { get; set; }

            [ColumnName("labels")]
            public long[] Labels { get; set; }

            [ColumnName("boxes")]
            public float[] Boxes { get; set; }

            [ColumnName("scores")]
            public float[] Scores { get; set; }
        }

        private class INPUT
        {
            public const int ChannelAmount = 3;
            public const int Width = 300;
            public const int Height = 300;

            [ColumnName("input")]
            [VectorType(1, ChannelAmount, Width, Height)]
            public float[] Values { get; set; }
        }

        private readonly PredictionEngine<INPUT, OUTPUT> predictionEngine;

        public Detector()
        {
            var mlContext = new MLContext();

            var gpuModelEstimator = mlContext.Transforms.ApplyOnnxModel(outputColumnNames: new string[] { "masks", "labels", "boxes", "scores" }, inputColumnNames: new string[] { "input" }, modelFile: "model.onnx", gpuDeviceId: 0);

            var data = mlContext.Data.LoadFromEnumerable(new List<INPUT>());

            var gpuModel = gpuModelEstimator.Fit(data);

            predictionEngine = mlContext.Model.CreatePredictionEngine<INPUT, OUTPUT>(gpuModel, ignoreMissingColumns: false);
        }

        private static float[] ProcessImage(Vec3b[] rgbImage)
        {
            var pixcels = new List<float>();
            pixcels.AddRange(rgbImage.Select(pixcel => (float)pixcel.Item0 / 255));
            pixcels.AddRange(rgbImage.Select(pixcel => (float)pixcel.Item1 / 255));
            pixcels.AddRange(rgbImage.Select(pixcel => (float)pixcel.Item2 / 255));

            return pixcels.ToArray();
        }

        public IEnumerable<DetectionResult> DetectNext(Mat image)
        {
            Cv2.Resize(image.Clone(), image, new Size(INPUT.Width, INPUT.Height));            

            image.GetArray<Vec3b>(out var vectorizedImage);
            var pixcelsArray = ProcessImage(vectorizedImage);

            var input = new INPUT
            {
                Values = pixcelsArray
            };

            var result = predictionEngine.Predict(input);

            if(result is null)
            {
                return Enumerable.Empty<DetectionResult>();
            }

            var detectionResults = new List<DetectionResult>();
            for (int i = 0; i < result.Labels.Length; i++)
            {
                var mask = result.Masks.Skip(i * OUTPUT.Width * OUTPUT.Height)
                    .Take(OUTPUT.Width * OUTPUT.Height)
                    .ToArray();

                var label = result.Labels.Skip(i).FirstOrDefault();
                var coordinates = result.Boxes.Skip(i * 4).Take(4).ToArray();
                var score = result.Scores.Skip(i).FirstOrDefault();

                detectionResults.Add(
                    new DetectionResult()
                    {
                        Label = (DetectionLabel)label,
                        Box = new DetectionBox()
                        {
                            Point1 = new Point()
                            {
                                X = coordinates[0],
                                Y = coordinates[1]
                            },
                            Point2 = new Point()
                            {
                                X = coordinates[2],
                                Y = coordinates[3]
                            }
                        },
                        Score = score,
                        Mask = mask.Select(p => (byte)(p * 255)).Select(p => (byte)(p > 128 ? 255 : 0)).ToArray()
                    });
            }

            return detectionResults.AsEnumerable();
        }
    }
}
