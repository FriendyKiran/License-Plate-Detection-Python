import re
import cv2
import numpy
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'C:\Users\Kiran\AppData\Local\Programs\Tesseract-OCR\tesseract'
cascade = cv2.CascadeClassifier("haarcascade_russian_plate_number.xml")

#cap = cv2.VideoCapture('license_plate.mp4')
cap = cv2.VideoCapture('YouCut_20210717_105331927.mp4')
#cap = cv2.VideoCapture('YouCut_20210705_144658628.mp4')
#cap = cv2.VideoCapture('YouCut_20210705_145311291.mp4')
#cap = cv2.VideoCapture('YouCut_20210711_095008973.mp4')



while True:
    ret,img=cap.read()
    #print(ret)
    if ret:
        # color conversion
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        nplate = cascade.detectMultiScale(gray, 1.1, 4)
        for (a, b, c, d) in nplate:

            plate = img[b : b + d, a :a + c]
            cv2.imshow("result", plate)

            # grayscale conversion
            plate_gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)

            # resizing the image
            plate = cv2.resize(plate_gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
            cv2.imshow("gray", plate)

            plate = cv2.GaussianBlur(plate, (5, 5), 0)

            ret, thresh = cv2.threshold(plate, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
            cv2.imshow("thresh",thresh)
            rect_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))
            dilation = cv2.dilate(thresh, rect_kern, iterations=1)
            #cv2.imshow("dil", dilation)
            contours, hie = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            sorted_contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])
            im2 = plate.copy()

            for cnt in sorted_contours[1:5]:
                x, y, w, h = cv2.boundingRect(cnt)
                height, width = im2.shape
                if height / float(h) > 6: continue
                if width / float(w) > 15: continue
                area = height * width
                if area / (h * w) > 10: continue

                rect = cv2.rectangle(im2, (x, y), (x + w, y + h), (0, 255, 0), 2)
                if w > 20: w = w - 10
                roi = plate[y:y + h, x:x + w]
                cv2.imshow("roi",roi)
                roi = cv2.bitwise_not(roi)
                read = pytesseract.image_to_string(roi,config='-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ --psm 8 --oem 3')
                read = re.sub('[\W_]+', '', read)
                read = ''.join(e for e in read if e.isalnum())
                if len(read) > 4:
                    rect = cv2.rectangle(img, (a, b), (a + c, b + d), (0, 255, 0), 2)
                    cv2.putText(img, read, (a, b - 5), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 2)
                    print(read)
            # cv2.imshow("thres",im2)
        img = cv2.resize(img,None ,fx = 0.5,fy = 0.5,interpolation=cv2.INTER_CUBIC)
        cv2.imshow("result1", img)
        cv2.waitKey(1)
    else:
        break