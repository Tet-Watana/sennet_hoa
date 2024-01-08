import os

directory = "./data/blood_vessel_segmentation/val"
output_file = "./data/list/blood_vessel_seg/val.lst"
no_need_str = "./data/blood_vessel_segmentation/"

with open(output_file, "w") as file:
    for root, dirs, files in os.walk(directory):
        if "images" not in root:
            continue
        files.sort()  # Sort files in ascending order
        for filename in files:
            if filename.endswith(".tif"):
                # file.write(os.path.join(root, filename) + "\n")
                file_path = os.path.join(root, filename)
                out_img_str = file_path.replace(no_need_str, "")
                out_label_str = out_img_str.replace("images", "labels")
                if os.path.exists(os.path.join(no_need_str, out_label_str)):
                    file.write(out_img_str + "\t" + out_label_str + "\n")
                else:
                    file.write(out_img_str + "\n")
