from src.data_loader import DataLoader


def main():
    dl = DataLoader()
    data = dl.create_input_data()
    print(data.head(3))


if __name__ == '__main__':
    main()

