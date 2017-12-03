import numpy
import re
import os
import sys
import PIL.Image


classes_dict = {"aeroplane":0, "bicycle":1, "bird":2, "boat":3, "bottle":4, "bus":5, "car":6, "cat":7, "chair":8, "cow":9, "diningtable":10, "dog":11, "horse":12, "motorbike":13, "person":14, "pottedplant":15, "sheep":16, "sofa":17, "train":18, "tvmonitor":19}



# def convert(size, box):
#     dw = 1./size[0]
#     dh = 1./size[1]
#     x = (box[0] + box[1])/2.0
#     y = (box[2] + box[3])/2.0
#     w = box[1] - box[0]
#     h = box[3] - box[2]
#     x = x*dw
#     w = w*dw
#     y = y*dh
#     h = h*dh
#     return (x,y,w,h)


def convert_annotation(image_id):
    boxes = []
    
    file = open('train/{}'.format(image_id))
 
    lines = file.readlines()
    for line in lines:
        blob , i = str(line).split(".jpg_")
        ''' discard blob, is not needed'''
        parts = i.partition(',')
    
        if parts[2] not in "1 0\n": 
     
            strings = parts[2].split(" ")
    
            nums = [] 
            for item in strings:
                if len(item) is not 1:
                    nums.append(int(re.sub("[\s+]", "", item)))


            y_1 = nums[0] // 400
            y_2 = nums[len(nums) - 2] // 400
            x_1 = nums[0] % 400
            x_2 = x_1 + nums[1]

            #print("class {}, x_1 {}, x_2 {}, y_1 {}, y_2 {}".format(classes_dict[parts[0]],x_1,x_2,y_1,y_2))

            box = [classes_dict[parts[0]], x_1, y_1, x_2, y_2 ]

    boxes.append(box)

    return boxes



   #numpy.array()
              

            #out_file.write(str(parts[0]) + " " + " ".join([str(a) for a in bb]) + '\n')

files = os.listdir("train/")

text_files = []
image_files = []



for file in files:
	if ".jpg" in file:
		image_files.append(file)
	else:
		text_files.append(file)


annotations = []

print(len(image_files), len(text_files))
for file in text_files:
	annotations.append(convert_annotation(file))

image_labels = numpy.array(annotations)


images = []

for image in image_files:
	img = numpy.array(PIL.Image.open(os.path.join('train', image )).resize((400, 400)).convert('RGB'), dtype=numpy.uint8)
	images.append(img)

images = numpy.array(images, dtype=numpy.uint8)


print("Dataset contains {} images".format(images.shape[0]))
numpy.savez("nc16_dataset", images=images, boxes=image_labels)
