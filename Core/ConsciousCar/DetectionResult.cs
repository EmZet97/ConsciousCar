namespace ConsciousCar
{
    public record DetectionResult
    {
        public DetectionLabel Label { get; init; }
        public DetectionBox Box { get; init; }
        public byte[] Mask { get; init; }
        public float Score { get; init; }
    }
}
