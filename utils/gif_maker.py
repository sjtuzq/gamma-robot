import imageio
import os

def compose_gif():
    img_paths = ["img/1.jpg","img/2.jpg","img/3.jpg","img/4.jpg"
    ,"img/5.jpg","img/6.jpg"]
    gif_images = []
    for path in img_paths:
        gif_images.append(imageio.imread(path))
    imageio.mimsave("test.gif",gif_images,fps=1)


def my_compose(img_path):
    gif_images = []
    img_list = os.listdir(img_path)
    img_list = sorted(img_list,key=lambda x:int(x.split('.')[0]))[:8]
    for file in img_list:
        gif_images.append(imageio.imread(os.path.join(img_path,file)))
    imageio.mimsave(os.path.join(img_path,"test.gif"),gif_images,fps=2)

def my_compose2(img_path):
    gif_images = []
    img_list = sorted(os.listdir(img_path),key=lambda x:int(x.split('_')[1].split('.')[0]))
    for file in img_list:
        gif_images.append(imageio.imread(os.path.join(img_path,file)))
    imageio.mimsave(os.path.join(img_path,"test.gif"),gif_images,fps=10)


class GIF:
    def __init__(self,test_id,img_num=9):
        self.test_id = test_id
        self.img_num = img_num
        self.root_path = './gif/{}'.format(self.test_id)

    def run(self):
        for folder in os.listdir(self.root_path):
            if '.gif' in folder:
                continue
            img_path = os.path.join(self.root_path,folder)
            img_list = os.listdir(img_path)
            img_list = sorted(img_list,key=lambda x:int(x.split('.')[0]))[:self.img_num]
            gif_images = []
            for file in img_list:
                gif_images.append (imageio.imread (os.path.join (img_path, file)))
            output_path = os.path.join(self.root_path,folder+'.gif')
            imageio.mimsave (output_path, gif_images, fps=3)

if __name__ == '__main__':
    # img_path = './gif/31_left'
    # img_path = './gif/31_right'
    # img_path = './gif/31_up'
    # img_path = './gif/31_down'
    # my_compose(img_path)

    # for file in os.listdir('./gif'):
    #     try:
    #         test_id = int(file)
    #     except:
    #         print('{} is not a valid experiment log folder!'.format(file))
    #         continue
    #     agent = GIF(test_id)
    #     agent.run()
    #     print(file)

    # my_compose('./gif/47/hold1')
    # my_compose('./gif/47/hold2')

    # my_compose2('./logs/before')
    # my_compose2('./logs/after')

    my_compose('./gif_final/epoch-8')