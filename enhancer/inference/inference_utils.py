from enhancer import *
from enhancer.utils import *
from .inference_helpers import *

def convert_prediction(arr,filename=None,denormalize=False):
    arr.clamp_(0,1)
    a = im_convert(arr[0],denormalize=denormalize)
    im = Image.fromarray(np.uint8(a*255))
    if filename:
        plt.imsave(filename+'.png', a)
    return im

# def predict_without_patch(filename,generator,scale=4,resize=True,profile=False):
#     bicubic = None
#     if type(filename) == type(""):
#         ex = Image.open(filename).convert("RGB")
#     else:
#         ex = filename.copy()
#         filename = "sample.png"
#         ex_img = ex.copy()
#     if resize:
#         ex= ex.resize((ex.size[0]//scale,ex.size[1]//scale))
#         bicubic = ex.resize((ex.size[0]*scale,ex.size[1]*scale),resample = Image.BICUBIC)
#         ex_img = ex.copy()
#     ex = np.array(ex).transpose(2,0,1)/255.
#     ex = torch.from_numpy(ex).unsqueeze(0).float().to(device)
#     with torch.no_grad():
#         generator.eval()
#         start=time.time()
#         result = generator(ex)
#         if profile==True:
#             print(time.time()-start)
#     img = convert_prediction(result)
#     if resize ==False:
#         bicubic = ex_img.resize((ex_img.size[0]//scale,ex_img.size[1]//scale))
#     return img,ex_img,bicubic

def predict_without_patch(image,generator,scale=4,resize=True,profile=False):
    bicubic = None
    if type(image) == type(""):
        image = Image.open(image).convert("RGB")
    if resize:
        ex_image = image.copy()
        image = image.resize((image.size[0]//scale,image.size[1]//scale))
        bicubic = image.resize((image.size[0]*scale,image.size[1]*scale),resample = Image.BICUBIC)
    else:
        bicubic = image.resize((image.size[0]*scale,image.size[1]*scale))
        ex_image = image.copy()
    image = np.array(image).transpose(2,0,1)/255.
    image = torch.from_numpy(image).unsqueeze(0).float().to(device)
    with torch.no_grad():
        generator.eval()
        start=time.time()
        result = generator(image)
        if profile:
            print(time.time()-start)
    result_image = convert_prediction(result)
    return result_image,ex_image,bicubic

def patch_prediction(img,generator,patch_size=150,scale_factor=4):
    try:
        img_splitter = ImageSplitter(seg_size=patch_size, scale_factor=scale_factor, boarder_pad_size=1)
        img_patches = img_splitter.split_img_tensor(img, scale_method=None, img_pad=0)
        out=[]
        with torch.no_grad():
         for j,i in enumerate(tqdm(img_patches)):
            with torch.no_grad():
                generator.eval()
                out.append(generator(i.to(device)))
        img_upscale = img_splitter.merge_img_tensor(out)
        a=convert_prediction(img_upscale)
        return a
    except Exception as e:
        print(str(e))
        return False

def predict_with_patch(image,generator,scale=4,resize=False,patch_size=150):
    bicubic = None
    if type(image) == type(""):
        ex = Image.open(image).convert("RGB")
    else:
        ex = image.copy()
        ex_img = image.copy()
    if resize:
        ex= ex.resize((ex.size[0]//scale,ex.size[1]//scale))
        bicubic = ex.resize((ex.size[0]*scale,ex.size[1]*scale),resample = Image.BICUBIC)
        ex_img = ex.copy()
    img = patch_prediction(ex,generator,patch_size=patch_size,scale_factor=scale)
    if resize ==False:
        bicubic = ex_img.resize((ex_img.size[0]//scale,ex_img.size[1]//scale))
    return img,ex_img,bicubic


def extractImages(pathIn, pathOut="./result.mp4",pattern="frame",dest="jpg",verbose=True,break_step=-1):
    count = 0
    if not os.path.exists(pathOut):
        os.mkdir(pathOut)
    vidcap = cv2.VideoCapture(pathIn)
    success, image = vidcap.read()
    success = True
    while success:
        success, image = vidcap.read()
        if verbose:
            print("Read a new frame: ", success)
            print(count)
        if success == True:
            cv2.imwrite(
                f"{pathOut}/{pattern}{count}.{dest}" ,image
            )  # save frame as JPEG file
        count = count + 1
        if count == break_step:
            break


def write_video_opencv(images,video_name,fps=25,fourcc=cv2.VideoWriter_fourcc(*'MJPV'), verbose = True ):
    frame = cv2.imread(images[0])
    height, width, layers = frame.shape
    video = cv2.VideoWriter(video_name,fourcc, fps, (width, height))  
    for image in images:
        if verbose:
            print(image)
        video.write(cv2.imread(image))
    cv2.destroyAllWindows()
    video.release()

def predict_video_frames(pathIn,pathOut="./final",predict_with_path=True,scale=4,resize=False,patch_size=150,verbos=True):
    start = time.time()
    if not os.path.exists(pathOut):
        os.mkdir(pathOut)
    for i in os.listdir(pathIn):
        path = f"{pathIn}/{i}"
        if verbose:
            print(i)
        if predict_with_path==True:
            result = predict_with_patch(path, generator,scale=scale,resize=resize,patch_size=patch_size)
        else:
            result =  predict_without_patch(path,scale=scale,resize=resize)
        result.save( f"{pathOut}/{i}")
    end = time.time()
    print(end - start)

def get_fps(filename):
    video = cv2.VideoCapture(filename)
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
    if int(major_ver)  < 3 :
        fps = video.get(cv2.cv.CV_CAP_PROP_FPS)
    else :
        fps = video.get(cv2.CAP_PROP_FPS)
    video.release(); 
    return fps


    
