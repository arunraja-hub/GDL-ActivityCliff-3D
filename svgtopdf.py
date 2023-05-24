from svglib.svglib import svg2rlg
from reportlab.graphics import renderPDF
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--file', type=str, help='Description of arg1')
args = parser.parse_args()


drawing = svg2rlg(str(args.file))
renderPDF.drawToFile(drawing, args = parser.parse_args()[:-4]+".pdf")