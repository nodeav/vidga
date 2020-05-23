namespace vidga {
    class SerDes {
    public:
        std::vector<uint8_t> serialize(std::shared_ptr<simplePopulation>);
        std::shared_ptr<simplePopulation> deserialize(std::vector<uint8_t>);
    };
}