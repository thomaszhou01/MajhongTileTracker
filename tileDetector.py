import numpy as np
import cv2
import pyautogui


def findTileMatch(tile, compareList):
    count = float("inf")
    lowest = compareList[0]

    for i in compareList:
        compare = tile
        compare2 = i[1]

        newCompare = cv2.resize(compare, (200, 400))
        newCompare2 = cv2.resize(compare2, (200, 400))

        diff = cv2.absdiff(newCompare2, newCompare)

        diffPix = np.sum(diff > 10)

        if diffPix < count:
            count = diffPix
            lowest = i
    cv2.imshow("l4", lowest[1])
    return lowest


def getContours(image):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(img_gray, 150, 255, cv2.THRESH_BINARY)

    # detect the contours on the binary image
    contours, _ = cv2.findContours(
        image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE
    )

    if len(contours) == 0:
        return []

    # filter small contours and approx their shapes
    selected_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 6000:
            epsilon = 0.1 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            coords = approx.reshape(4, 2)
            selected_contours.append(coords)
    return selected_contours


def getTile(image, contours):
    # sort contours and determine if a new tile is drawn
    contours.sort(key=lambda x: x[0][0])

    coords = contours[-1]
    coordsSorted = sorted(coords, key=lambda x: (x[0], x[1]))

    coords2 = contours[-2]
    coordsSorted2 = sorted(coords2, key=lambda x: (x[0], x[1]))

    if coordsSorted[0][0] - coordsSorted2[2][0] < 10:
        return np.array([])

    # get the drawn tile and return
    tile = image.copy()
    x1 = coordsSorted[1][0]
    x2 = coordsSorted[3][0]
    y1 = coordsSorted[1][1]
    y2 = coordsSorted[2][1]
    tile = tile[y1:y2, x1:x2]
    tile = cv2.resize(tile, (200, 400), interpolation=cv2.INTER_LINEAR)
    tileGrey = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)
    _, tileThresh = cv2.threshold(tileGrey, 150, 255, cv2.THRESH_BINARY)

    # Test drawing
    image_copy = image.copy()
    cv2.drawContours(
        image=image_copy,
        contours=contours,
        contourIdx=-1,
        color=(0, 255, 0),
        thickness=2,
        lineType=cv2.LINE_AA,
    )
    cv2.imshow("l1", image_copy)
    # image_copy2 = image.copy()
    # cv2.drawContours(
    #     image=image_copy2,
    #     contours=[coords],
    #     contourIdx=-1,
    #     color=(0, 255, 0),
    #     thickness=2,
    #     lineType=cv2.LINE_AA,
    # )
    # cv2.imshow("l2", image_copy2)
    # cv2.imshow("l3", tileThresh)

    return tileThresh


def runDetection(compare, region):
    cv2.namedWindow("l1", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("l1", 700, 400)

    started = False
    found = 0
    while True:
        img = pyautogui.screenshot(region=region)

        frame = np.array(img)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        try:
            contours = getContours(frame)
            if len(contours) == 14:
                started = True
            elif len(contours) == 0:
                started = False

            if started:
                tile = getTile(frame, contours)
                if tile.size != 0 and found < 20:
                    identified = findTileMatch(tile, compare)
                    found += 1
                    if found == 20:
                        yield identified[0]
                elif tile.size == 0:
                    found = 0

        except Exception as e:
            print(e)
            continue

        if cv2.waitKey(1) == ord("q"):
            break

    cv2.waitKey(0)

    cv2.destroyAllWindows()
