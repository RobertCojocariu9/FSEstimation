from models.base_model import BaseModel
from models.fsnet_model import FSNetModel


def create_model(opt, dataset):
    """
    Instantiates the model, based on the specified model in the options
    :param opt: BaseOptions object
    :param dataset: Dataset object
    :return: Instantiated model
    """
    if opt.model == "fsnet":
        model = FSNetModel(opt, dataset)
    else:
        print("Model [%s] does not exist" % opt.model)
        exit(1)
    print("Model [%s] was created" % model.__class__.__name__)
    return model
