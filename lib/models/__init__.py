from models.layer import make_linear_layers, make_conv_layers, make_deconv_layers, LocallyConnected2d
from models.resnet import PoseResNet
from models.hrnet import PoseHighResolutionNet
<<<<<<< HEAD
from models.module import Projector, Predictor, BodyPredictor, HeatmapPredictor
=======
from models.module import Projector, BodyProjector, FCBodyPredictor, BodyPredictor, HeatmapPredictor
>>>>>>> e626c3b948bf2c5179adc2c9779ea41d243eeaaf
from models.model import get_model, transfer_backbone