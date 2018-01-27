import Extractor


def crop_image(image):
    matrix = Extractor.ImageToMatrix(image)
    # print(matrix)
    top,bottom,left,right = get_image_bounds(matrix)
    # print(get_image_bounds(matrix))
    img = image.crop((left, top, right + 1, bottom + 1))
       
    return img  


def get_image_bounds(matrix):
    LIMIT = 200
    # Top bound
    top = 69
    found = False
    i = 0
    while found == False:
        for x in range(len(matrix[i])):
            if matrix[i][x] < LIMIT:
                top = i
                found = True
        i += 1
    
    # Bottom bound
    bottom = 69
    found = False
    i = 27
    while found == False:
        for x in range(len(matrix[i])):
            if matrix[i][x] < LIMIT:
                bottom = i
                found = True
        i -= 1
    
    # Left bound
    left = 69
    found = False
    i = 0
    while found == False:
        for x in range(len(matrix[i])):
            if matrix[x][i] < LIMIT:
                left = i
                found = True
        i += 1
       
    # Right bound
    right = 69
    found = False
    i = 27
    while found == False:
        for x in range(len(matrix[i])):
            if matrix[x][i] < LIMIT:
                right = i
                found = True
        i -= 1

    return (top, bottom, left, right)
