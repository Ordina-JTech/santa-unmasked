package nl.ordina.jtech.deeplearning.santa.handwriting;

import java.io.IOException;
import java.util.List;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.core.io.Resource;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestParam;

import nl.ordina.jtech.deeplearning.santa.handwriting.model.Prediction;
import nl.ordina.jtech.deeplearning.santa.handwriting.storage.StorageService;

@Controller
public class UploadController {

    private final StorageService storageService;
    private final SantaHandwritingClassifier mnistClassifier;
    
    @Autowired
    public UploadController(StorageService storageService, SantaHandwritingClassifier mnistClassifier) {
        this.storageService = storageService;
        this.mnistClassifier = mnistClassifier;
    }

    @GetMapping("/")
    public String listUploadedFiles(Model model) throws IOException {
        return "mnistNumbers";
    }

    @PostMapping("/upload/drawing")
    public String handleWebcamUpload(@RequestParam("imgBase64") String data, Model model) {

        String base64Image = data.split(",")[1];
        storageService.store(base64Image);

        Resource image = storageService.loadAsResource("mnist.jpg");
        List<Prediction> predictions = null;
        try {
            predictions = mnistClassifier.classify(image.getInputStream());
        } catch (Exception e) {
            e.printStackTrace();
        }

        model.addAttribute("image", "/files/mnist.jpg");
        model.addAttribute("predictions", predictions);
        model.addAttribute("message", "You successfully uploaded a mnist image!");

        return "mnistNumbers :: predictions";

    }

    @ExceptionHandler(Exception.class)
    public ResponseEntity handleStorageFileNotFound(Exception exc) {
        return ResponseEntity.notFound().build();
    }

}
