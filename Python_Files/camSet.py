import cv2
from vimba import *
with Vimba.get_instance () as vimba:
    cams = vimba.get_all_cameras ()

with cams [0] as cam:
    frame = cam.get_frame ()
    frame.convert_pixel_format(PixelFormat.Mono8)
    cv2.imwrite('frame.jpg', frame.as_opencv_image ())


# from vimba import *
# with Vimba.get_instance () as vimba:
# cams = vimba.get_all_cameras (1)

# from vimba import *

# with Vimba.get_instance () as vimba:
#     inters = vimba.get_all_interfaces ()
#     with inters [0] as interface:
#         for feat in interface.get_all_features ():
#             print(feat)
            
            
# # Synchronous grab
# from vimba import *

# with Vimba.get_instance () as vimba:
#     cams = vimba.get_all_cameras ()
#     with cams [0] as cam:
#         # Aquire single frame synchronously
#         frame = cam.get_frame ()

#         # Aquire 10 frames synchronously
#         for frame in cam.get_frame_generator(limit =10):
#             pass