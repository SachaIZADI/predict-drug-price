from src.train import train
from src.predict import predict
import logging
import click

logging.basicConfig(level=logging.INFO)


# TODO :
#  - [ ] Make a log model
#  - [ ] Update readme
#    - [ ] How to run
#    - [ ] Modelling steps
#    - [ ] Caveats & next steps


@click.command()
@click.option("--do-train", is_flag=True)
@click.option("--do-predict", is_flag=True)
def main(do_train: bool, do_predict: bool):
    if do_train:
        train()
    if do_predict:
        predict()


if __name__ == "__main__":
    main()
