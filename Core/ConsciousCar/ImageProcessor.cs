using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Linq;

namespace ConsciousCar;

public class ImageProcessor
{
    record MaskColor(byte Red, byte Green, byte Blue);

    record LabelWithMask(DetectionLabel label, MaskColor Color);

    public int ImageHeight { get; }
    public int ImageWidth { get; }

    public ImageProcessor(int imageHeight, int imageWidth)
    {
        ImageHeight = imageHeight;
        ImageWidth = imageWidth;
    }

    private Mat GenerateMasks(IEnumerable<DetectionResult> detections, DetectionLabel[] labelIds)
    {
        var fin_mask = new Mat(ImageHeight, ImageWidth, MatType.CV_8UC1, Enumerable.Repeat((byte)0, ImageHeight * ImageWidth).ToArray());
        foreach (var detectionResult in detections)
        {
            if (!labelIds.Contains(detectionResult.Label)) continue;
            var temporary_mask = detectionResult.Mask.Clone();
            Cv2.AddWeighted(temporary_mask.Clone(), 3, temporary_mask.Clone(), 0, -128 * 3, temporary_mask);

            Cv2.AddWeighted(fin_mask.Clone(), 1, temporary_mask.Clone(), 1, 0, fin_mask);
        }

        return fin_mask;
    }

    private Mat GenerateBoundingBoxes(Mat inputImage, IEnumerable<DetectionResult> detections, LabelWithMask[] labelWithMasks)
    {
        foreach (var detectionResult in detections)
        {
            if (!labelWithMasks.Select(x => x.label).Contains(detectionResult.Label)) continue;
            Cv2.Rectangle(inputImage,
                new OpenCvSharp.Point(detectionResult.Box.Point1.X, detectionResult.Box.Point1.Y),
                new OpenCvSharp.Point(detectionResult.Box.Point2.X, detectionResult.Box.Point2.Y),
                new Scalar(0, 255, 0),
                thickness: 1,
                lineType: LineTypes.Link4);

            var textColor = labelWithMasks.SingleOrDefault(x => x.label == detectionResult.Label).Color;

            Cv2.PutText(inputImage,
                $"{detectionResult.Label}",// {detectionResult.Score}",
                new OpenCvSharp.Point(detectionResult.Box.Point1.X + 5, detectionResult.Box.Point1.Y + 15),
                HersheyFonts.Italic,
                0.5,
                new Scalar(0, 0, 255),
                thickness: 2);
        }

        return inputImage;
    }

    private Mat ConvertToColorMask(Mat grayscaleInputMask, MaskColor color)
    {
        var colorDensity = 2;
        var colorData = Enumerable
            .Range(0, ImageHeight * ImageWidth * 3)
            .Select((x, y) => x % (3 * colorDensity) == 0 || (x / ImageWidth) % colorDensity == 0 && x % 3 == 0 ? (byte)color.Blue
                            : x % (3 * colorDensity) == 1 || (x / ImageWidth) % colorDensity == 0 && x % 3 == 1 ? (byte)color.Green
                            : x % (3 * colorDensity) == 2 || (x / ImageWidth) % colorDensity == 0 && x % 3 == 2 ? (byte)color.Red
                            : (byte)0
                    )
            .ToArray();

        var colorMask = new Mat(ImageHeight, ImageWidth, MatType.CV_8UC3, colorData);

        var mask_colored = new Mat();

        Cv2.CvtColor(grayscaleInputMask.Clone(), mask_colored, ColorConversionCodes.GRAY2BGR);

        Cv2.BitwiseAnd(mask_colored, colorMask, colorMask);
        return colorMask;
    }

    public Mat GenerateNext(Mat inputImage, IEnumerable<DetectionResult> detections)
    {
        var outputImage = inputImage.Clone();

        var detectionDefinitions = new[] {
            new LabelWithMask(DetectionLabel.Road, new MaskColor(0,0,255)),
            new LabelWithMask(DetectionLabel.Sidewalk, new MaskColor(255,20,147)),
            new LabelWithMask(DetectionLabel.Vehicle, new MaskColor(255,69,0)),
            new LabelWithMask(DetectionLabel.Person, new MaskColor(255,0,0)),
            new LabelWithMask(DetectionLabel.TrafficLight, new MaskColor(255,0,0)),
            new LabelWithMask(DetectionLabel.TrafficSign, new MaskColor(255,255,0)),
            new LabelWithMask(DetectionLabel.Background, new MaskColor(0,255,0)),
            new LabelWithMask(DetectionLabel.Building, new MaskColor(64,64,64)),
        };

        foreach (var label in detectionDefinitions)
        {
            var mask = GenerateMasks(detections, labelIds: new[] { label.label });
            var met = ConvertToColorMask(mask.Clone(), label.Color);
            Cv2.AddWeighted(outputImage.Clone(), 1, met.Clone(), 0.3, 0, outputImage);
        }

        GenerateBoundingBoxes(outputImage, detections, detectionDefinitions);
        return outputImage;
    }
}
