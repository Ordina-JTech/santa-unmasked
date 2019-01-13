package nl.ordina.jtech.deeplearning.santa.unmasked.storage;

class StorageFileNotFoundException extends StorageException {

    StorageFileNotFoundException(String message) {
        super(message);
    }

    StorageFileNotFoundException(String message, Throwable cause) {
        super(message, cause);
    }
}
