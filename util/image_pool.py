import random
import torch


"""
1. def query 에서 왜 image를 unsqueeze 하지..?
2. else: --- 이미지 풀 다 찼을때 동작 잘 모르겠다...
4. torch.cat 쓰는 이유 torch.stack 이 더 깔끔할 것 같은데
"""
class ImagePool():
    """
    this class implements an image buffer that stores previously generated images
    """
    def __init__(self, pool_size):
        self.pool_size = pool_size
        # 빈 image pool 만들기
        if self.pool_size > 0:
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        """return an image from the pool"""
        if self.pool_size == 0:
            return images
        return_images = []
        for image in images:
            image = torch.unsqueeze(image.data, 0)
            if self.num_imgs < self.pool_size:
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else: # image pool 에 이미지 다 차있으면
                p = random.uniform(0,1)
                if p>0.5:
                    random_id = random.randint(0, self.pool_size - 1)
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else: #p<0.5
                    return_images.append(image)
        return_images = torch.cat(return_images, 0)
        return return_images
