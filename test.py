train_img_array = list([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
# test_img_array = list([11, 12, 13, 14, 15, 16, )

# Check the number of type image extension in the dataset
train_image_types = [
    {"file_type": "jpeg", "count": 0},
    {"file_type": "png", "count": 0},
    {"file_type": "jpg", "count": 0},
    {"file_type": "other", "count": 0},
]

for file_name in train_img_array:
    # Split the file name by the period (.) character
    parts = str(file_name).split(".")
    file_extension = str(parts[-1]).lower()
    found = False
    for counter in range(len(train_image_types)):
        if file_extension == train_image_types[counter]["file_type"]:
            train_image_types[counter]["counter"]=+1
            found = True
            break
        if train_image_types[counter]["file_type"] == "other":
            train_image_types[counter]["counter"]=+1
            break
    
    
print(train_image_types)