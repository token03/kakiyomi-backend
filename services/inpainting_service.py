import cv2

from services.inpainting.lama import LaMa
from services.inpainting.mi_gan import MIGAN
from services.inpainting.schema import Config

inpaint_map = {
    "LaMa": LaMa,
    "MI-GAN": MIGAN
}

DEFAULT_INPAINTER = "LaMa"
class InpaintingService:
    def __init__(self):
        print("Initializing Inpainting Service")

        self.device = 'cpu' # TODO: add gpu support later
        InpainterClass = inpaint_map[DEFAULT_INPAINTER]
        self.inpainter = InpainterClass(self.device)
        self.config = Config()

        print("Inpainting Service initialized")

    def inpaint(self, image, mask):
        inpainted = self.inpainter(image, mask, self.config)
        return cv2.convertScaleAbs(inpainted)