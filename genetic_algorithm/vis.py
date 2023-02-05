# https://github.com/andrewdcampbell/seam-carving
def visualize(image, bool_mask=None):
    display = image.astype(np.uint8)

    if bool_mask is not None:
        display[np.where(bool_mask == False)] = np.array([0, 0, 255])

    # display_resize = cv2.resize(display, (1000, 500))
    # cv2.imshow("visualization", display_resize)
    cv2.imshow("visualization", display)
    cv2.waitKey(100)

    return display

def visualize_seams(image, original_mask=None):
    display = image.astype(np.uint8)

    if original_mask is not None:
        display[np.where(original_mask < 0)] = np.array([0, 0, 255])
    return display