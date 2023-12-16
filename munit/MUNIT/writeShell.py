import os

if __name__ == "__main__":
    f = open("scripts/sunny2snow_night.cmd", "a")
    order = "python ../test.py --config ../configs/snow_night.yaml --output_folder ../outputs/snow_night --checkpoint ../models/snow_night.pt --a2b 1 --num_style 1"
    for img_name in os.listdir("E:\\Lab\\research\\MUNIT\\inputs\\sunny"):
        f.write(order + " --input E:\\Lab\\research\\MUNIT\\inputs\\sunny\\" + img_name + "\n")
    f.close()
