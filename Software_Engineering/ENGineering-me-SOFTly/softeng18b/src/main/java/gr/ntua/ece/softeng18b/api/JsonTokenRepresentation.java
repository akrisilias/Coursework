package gr.ntua.ece.softeng18b.api;

import com.google.gson.Gson;
import gr.ntua.ece.softeng18b.data.model.Token;
import org.restlet.data.MediaType;
import org.restlet.representation.WriterRepresentation;

import java.io.IOException;
import java.io.Writer;

public class JsonTokenRepresentation extends WriterRepresentation {

    private final Token token;

    public JsonTokenRepresentation(Token token) {
        super(MediaType.APPLICATION_JSON);
        this.token = token;
    }

    @Override
    public void write(Writer writer) throws IOException {
        Gson gson = new Gson();
        writer.write(gson.toJson(token));
    }
}
