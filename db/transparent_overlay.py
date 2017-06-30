import numpy as np
import cv2

def overlay_text(image):
    # loop over the alpha transparency values
    for alpha in np.arange(0, 1.1, 0.1)[::-1]:
        # create two copies of the original image -- one for
        # the overlay and one for the final output image
        overlay = image.copy()
        output = image.copy()
        cv2.putText(overlay, "PyImageSearch: alpha={}".format(alpha),
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)
        return output
