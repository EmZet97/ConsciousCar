namespace ConsciousCar
{
    public record Point
    {
        public float X { get; init; }
        public float Y { get; init; }
    }
    public record DetectionBox
    {
        public Point Point1 { get; init; }
        public Point Point2 { get; init; }
    }
}
