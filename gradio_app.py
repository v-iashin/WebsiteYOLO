from pathlib import Path

import torch
import gradio as gr
from glob import glob

import sys
sys.path.insert(0, './WebsiteYOLO')

from darknet import Darknet
from utils import check_if_file_exists_else_download, predict_and_save

class App:

    def __init__(self,
            weights_path='./weights/yolov3.weights',
            config_path='./cfg/yolov3_608x608.cfg',
            labels_path='./data/coco.names',
            font_path='./data/FreeSansBold.ttf',
            examples_glob="./data/*.jpg",
            **gr_interface_kwargs,
        ) -> None:
        self.device = torch.device('cpu')
        self.weights_path = Path(weights_path)
        self.config_path = Path(config_path)  # yolov3_416x416.cfg also available
        self.labels_path = Path(labels_path)
        self.font_path = Path(font_path)
        self.examples = sorted(glob(examples_glob), reverse=True)

        self.model = Darknet(self.config_path)
        self.model.load_weights(check_if_file_exists_else_download(self.weights_path))
        self.model.eval()

        self.iface = gr.Interface(
            fn=self.predict,
            inputs=gr.Image(type='pil'),
            outputs='image',
            examples=self.examples,
            **gr_interface_kwargs,
        )
        self.iface.launch()

    def predict(self, source_img):
        with torch.no_grad():
            predictions, img = predict_and_save(
                source_img, self.model, self.device, self.labels_path, self.font_path, orientation=None
            )
        return img

if __name__ == '__main__':
    App()
