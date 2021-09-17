namespace ConsciousCar
{
    public record DetectionResult
    {
        public int Label { get; init; }
        public DetectionBox Box { get; init; }
        public byte[] Mask { get; init; }
        public float Score { get; init; }
    }
}
