package nl.ordina.jtech.deeplearning.santa.unmasked;

import java.io.IOException;
import java.util.ArrayList;

import nl.ordina.jtech.deeplearning.santa.unmasked.model.Prediction;
import nl.ordina.jtech.deeplearning.santa.unmasked.storage.StorageService;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.core.io.Resource;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.ResponseBody;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.web.servlet.mvc.support.RedirectAttributes;

@Controller
public class UploadController {

    private final StorageService storageService;
    private SantaClassifier imageClassifier;

    @Autowired
    public UploadController(StorageService storageService, SantaClassifier imageClassifier) {
        this.storageService = storageService;
        this.imageClassifier = imageClassifier;
    }

    @GetMapping("/")
    public String listUploadedFiles(Model model) throws IOException {
        return "uploadForm";
    }

    @PostMapping("/upload/webcam")
    public String handleWebcamUpload(@RequestParam("imgBase64") String data, Model model) {

        String base64Image = data.split(",")[1];
        storageService.store(base64Image);

        Resource image = storageService.loadAsResource("webcam.jpg");
        ArrayList<Prediction> predictions = null;
        try {
            predictions = imageClassifier.classify(image.getInputStream());
        } catch (IOException e) {
            e.printStackTrace();
        }

        model.addAttribute("image", "/files/webcam.jpg");
        model.addAttribute("predictions", predictions);
        model.addAttribute("message", "You successfully uploaded a webcam image!");

        return "uploadForm :: predictions";

    }

    @ExceptionHandler(Exception.class)
    public ResponseEntity handleStorageFileNotFound(Exception exc) {
        return ResponseEntity.notFound().build();
    }

}
