// Elements for taking the snapshot
var canvas = document.getElementById('canvas');
var video = document.getElementById('video');
let videoCapture = new cv.VideoCapture(video);
// source image element matrix
let src = new cv.Mat(240, 320, cv.CV_8UC4);
// destination matrix, will act as a copy so we dont change the source
let dst = new cv.Mat(240, 320, cv.CV_8UC4);
// will contain all the detected faces in the picture
let faces = new cv.RectVector();
// the actual face detector/classifier
let classifier = new cv.CascadeClassifier();

/**
*	Loads a pre-trained face-detection haarcascade file to our classifier. 
*/
function loadHaarCascadeFile() {
	let utils = new Utils('errorMessage');
	let faceCascadeFile = 'haarcascade_frontalface_default.xml';
	utils.createFileFromUrl(faceCascadeFile, faceCascadeFile, () => {
	    if(classifier.load(faceCascadeFile)) {
	    } else {
	        console.log("cascade file not loaded");
	    }
	});
}

/**
*	Request access to the webcam and show stream
*/
function getAccessToTheCamera() {
	if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
	    // Not adding `{ audio: true }` since we only want video now
	    navigator.mediaDevices.getUserMedia({video: true}).then(function (stream) {
	        try {
	            video.srcObject = stream;
	        } catch (error) {
	            video.src = window.URL.createObjectURL(stream);
	        }
	        video.play();
	    });
	}
}

function processVideo() {
    try{ 
        // start processing the current videoframe
        videoCapture.read(src);
        src.copyTo(dst);
        // detect faces.
        
        classifier.detectMultiScale(dst, faces, 1.1, 3, 0);
        // take the first/biggest face
        let face = faces.get(0);
        // add offset for the hat, just in case santa passes by.
        let santaHatOffsetHeight = face.height;
        // if starting position will be below 0, set it to 0. Prevents the image from having empty black borders
        let startY = face.y - santaHatOffsetHeight < 0 ? 0 : face.y - santaHatOffsetHeight;
        let height = face.height + santaHatOffsetHeight;
        // crop the image to only contain the face including our (possible) santa hat
        let rect = new cv.Rect(face.x, startY, face.width, height);
        dst = src.roi(rect);
        cv.imshow('canvas', dst);
        postImage();
    }catch(ex) {
    	// prevents the front-end from crashing when no face is found
        console.error("no face found" + ex.message);
    }
   // schedule the next frame
   setTimeout(processVideo, 2000);
    
};

function postImage() {
    var params = canvas.toDataURL("image/jpeg", 0.85);
    $.ajax({
        type: "POST",
        url: "/upload/webcam",
        data: {
            contentType: "multipart/form-data",
            imgBase64: params,
            processData: false

        }
    }).done(function (predictions) {
        console.log('predictions' + predictions);
        $('#predictions').html(predictions);


    });
}

loadHaarCascadeFile();
getAccessToTheCamera();
setTimeout(processVideo, 200);

