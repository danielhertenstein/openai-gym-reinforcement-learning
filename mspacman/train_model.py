from mspacman.dqn import MspacmanDQN


def main():
    model = MspacmanDQN()
    model.train()


if __name__ == '__main__':
    main()
