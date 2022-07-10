using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ConsciousCar;

public class ImageProcessor
{
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
            var temporary_mask = detectionResult.Mask.Clone();
            Cv2.AddWeighted(temporary_mask.Clone(), 4, temporary_mask.Clone(), 0, -386, temporary_mask);

            //if (!labelIds.Contains(detectionResult.Label)) continue;
            Cv2.AddWeighted(fin_mask.Clone(), 1, temporary_mask.Clone(), 1, 0, fin_mask);
            //Cv2.BitwiseOr(fin_mask.Clone(), detectionResult.Mask, fin_mask);
        }

        return fin_mask;
    }

    private Mat GenerateBoundingBoxes(Mat inputImage, IEnumerable<DetectionResult> detections, DetectionLabel[] labelIds)
    {
        foreach (var detectionResult in detections)
        {
            //if (!labelIds.Contains(detectionResult.Label)) continue;
            Cv2.Rectangle(inputImage,
                new OpenCvSharp.Point(detectionResult.Box.Point1.X, detectionResult.Box.Point1.Y),
                new OpenCvSharp.Point(detectionResult.Box.Point2.X, detectionResult.Box.Point2.Y),
                new Scalar(0, 255, 0),
                thickness: 2);

            Cv2.PutText(inputImage,
                $"{detectionResult.Label} {detectionResult.Score}",
                new OpenCvSharp.Point(detectionResult.Box.Point1.X, detectionResult.Box.Point1.Y),
                new HersheyFonts(),
                0.5,
                new Scalar(255, 0, 255));
        }

        return inputImage;
    }

    private Mat ConvertToColorMask(Mat grayscaleInputMask, Scalar color)
    {
        var colorMask = new Mat(ImageHeight, ImageWidth, MatType.CV_8UC3, Enumerable.Range(0, ImageHeight * ImageWidth * 3).Select(x => x % 3 == 2 ? 255 : 0).ToArray());
        var mask_colored = new Mat();
        Cv2.CvtColor(grayscaleInputMask.Clone(), mask_colored, ColorConversionCodes.GRAY2BGR);

        Cv2.BitwiseAnd(mask_colored, colorMask, colorMask);
        return colorMask;
    }

    public Mat GenerateNext(Mat inputImage, IEnumerable<DetectionResult> detections)
    {
        //var fin_mask = new Mat(ImageHeight, ImageWidth, MatType.CV_8UC1, Enumerable.Repeat((byte)0, ImageHeight * ImageWidth).ToArray());
        var labelIds = new[] { DetectionLabel.Car };
        var mask = GenerateMasks(detections, labelIds: labelIds);

        var outputImage = inputImage.Clone();
        var met = ConvertToColorMask(mask.Clone(), new Scalar());
        Cv2.CvtColor(mask.Clone(), mask, ColorConversionCodes.GRAY2BGR);
        Cv2.AddWeighted(inputImage.Clone(), 1, met.Clone(), 0.5, 0, outputImage);

        //Cv2.BitwiseAnd(inputImage, mask, outputImage);//, mask: mask);
        GenerateBoundingBoxes(outputImage, detections, labelIds);
        return outputImage;
    }
}

