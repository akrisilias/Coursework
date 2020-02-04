package gr.ntua.ece.softeng18b.data.model;

import java.util.Objects;

public class Message {

    private final String message;

    public Message(String message) {
        this.message = message;
    }

    public String getMessage() {
        return message;
    }

}
