//("rtsp://streamuser:passwd123@10.159.120.250:1050/Streaming/Channels/602")
package main

import (
	"fmt"
	"image"
	"image/color"
	"os"

	"gocv.io/x/gocv"
)

func main() {
	// Check if the user provided an RTSP URL
	if len(os.Args) < 2 {
		fmt.Println("Please provide an RTSP URL as an argument")
		return
	}

	// Open the RTSP stream
	video, err := gocv.VideoCaptureFile(os.Args[1])
	if err != nil {
		fmt.Println(err)
		return
	}
	defer video.Close()

	// Load the YOLOv3 model
	net := gocv.ReadNet("yolov3.weights", "yolov3.cfg")
	if net.Empty() {
		fmt.Println("Error loading YOLOv3 model")
		return
	}
	defer net.Close()

	// Set the input size for the model
	net.SetInput(gocv.NewMatWithSize(1, 3, gocv.MatTypeCV8UC3), "")

	// Create a window to display the video
	window := gocv.NewWindow("YOLOv3")
	defer window.Close()

	// Loop through the video frames
	for {
		// Read the next frame
		mat := gocv.NewMat()
		if ok := video.Read(&mat); !ok {
			fmt.Println("Error reading video frame")
			return
		}
		if mat.Empty() {
			continue
		}

		// Detect people in the frame
		blob := gocv.BlobFromImage(mat, 1.0/255.0, image.Pt(416, 416), gocv.NewScalar(0, 0, 0, 0), true, false)
		net.SetInput(blob, "")
		prob := net.Forward("")
		blob.Close()

		// Extract the bounding boxes for each detected object
		boxes := gocv.GetBlobChannel(prob, 0, 0)
		defer boxes.Close()

		for i := 0; i < boxes.Total(); i += 4 {
			classID := int(boxes.GetFloatAt(i, 1))
			if classID != 0 { // 0 is the class ID for people
				continue
			}

			conf := boxes.GetFloatAt(i, 2)
			if conf < 0.5 { // Only draw boxes with confidence above 50%
				continue
			}

			x1 := int(boxes.GetFloatAt(i, 0)) * mat.Cols()
			y1 := int(boxes.GetFloatAt(i, 1)) * mat.Rows()
			x2 := int(boxes.GetFloatAt(i, 2)) * mat.Cols()
			y2 := int(boxes.GetFloatAt(i, 3)) * mat.Rows()


			// Draw a bounding box around the detected object
			gocv.Rectangle(&mat, image.Rect(x1, y1, x2, y2), color.RGBA{0, 255, 0, 0}, 3)
		}

		// Display the frame
		window.IMShow(mat)
		if window.WaitKey(1) >= 0 {
			break
		}
	}
}
