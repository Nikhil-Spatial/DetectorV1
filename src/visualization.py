import bbox_visualizer as bbv

def draw_rectangles(image, coords, labels):
    image = image.permute(1, 2, 0).numpy()
    image = bbv.draw_multiple_rectangles(image, coords, bbox_color=(255, 0, 0),
                                         thickness=1)
    image = bbv.add_multiple_labels(image, labels, coords)

    return image