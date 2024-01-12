import os
import argparse


class CreateList:
    def __init__(self, label_dir_path, out_lst_path, no_need_str):
        self.label_dir_path = label_dir_path
        self.out_lst_path = out_lst_path
        self.no_need_str = no_need_str

    def create_lst_file(self):
        with open(self.out_lst_path, "w") as file:
            for root, dirs, files in os.walk(self.label_dir_path):
                if "images" not in root:
                    continue
                files.sort()  # Sort files in ascending order
                for filename in files:
                    if filename.endswith(".tif"):
                        file_path = os.path.join(root, filename)
                        out_img_str = file_path.replace(self.no_need_str, "")
                        out_label_str = out_img_str.replace("images", "labels")
                        if os.path.exists(os.path.join(self.no_need_str, out_label_str)):
                            file.write(out_img_str + "\t" +
                                       out_label_str + "\n")
                        else:
                            file.write(out_img_str + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create list file")
    parser.add_argument("--label_dir_path", type=str, help="Label directory path",
                        default="data/blood_vessel_segmentation/test")
    parser.add_argument("--out_lst_path", type=str, help="Output lst file path",
                        default="data/list/blood_vessel_seg/test.lst")
    parser.add_argument("--no_need_str", type=str,
                        help="String to be replaced", default="data/blood_vessel_segmentation/")
    args = parser.parse_args()

    create_list = CreateList(
        args.label_dir_path, args.out_lst_path, args.no_need_str)
    create_list.create_lst_file()
