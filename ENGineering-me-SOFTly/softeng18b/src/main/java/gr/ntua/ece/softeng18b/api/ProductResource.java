package gr.ntua.ece.softeng18b.api;

import gr.ntua.ece.softeng18b.conf.Configuration;
import gr.ntua.ece.softeng18b.data.DataAccess;
import gr.ntua.ece.softeng18b.data.model.Product;
import gr.ntua.ece.softeng18b.data.model.Message;
import org.restlet.data.Status;
import org.restlet.representation.Representation;
import org.restlet.resource.ResourceException;
import org.restlet.resource.ServerResource;
import org.restlet.data.Form;

import org.restlet.util.Series;

import java.util.Optional;
import java.util.List;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Set;
import java.util.HashSet;

import javax.mail.Header;

public class ProductResource extends ServerResource {

    private final DataAccess dataAccess = Configuration.getInstance().getDataAccess();

    @Override
    protected Representation get() throws ResourceException {

        String idAttr = getAttribute("id");

        if (idAttr == null) {
            throw new ResourceException(Status.CLIENT_ERROR_BAD_REQUEST, "Missing product id");
        }

        Long id = null;
        try {
            id = Long.parseLong(idAttr);
        }
        catch(Exception e) {
            throw new ResourceException(Status.CLIENT_ERROR_BAD_REQUEST, "Invalid product id: " + idAttr);
        }

        Optional<Product> optional = dataAccess.getProduct(id);
        Product product = optional.orElseThrow(() -> new ResourceException(Status.CLIENT_ERROR_NOT_FOUND, "Product not found - id: " + idAttr));

        return new JsonProductRepresentation(product);
    }

    protected Representation put(Representation entity) throws ResourceException {

        String idAttr = getAttribute("id");

        if (idAttr == null) {
            throw new ResourceException(Status.CLIENT_ERROR_BAD_REQUEST, "Missing product id");
        }

        Long id = null;
        try {
            id = Long.parseLong(idAttr);
        }
        catch(Exception e) {
            throw new ResourceException(Status.CLIENT_ERROR_BAD_REQUEST, "Invalid product id: " + idAttr);
        }

        //Create a new restlet form
        Form form = new Form(entity);
        //Read the parameters
        String name = form.getFirstValue("name");
        String description = form.getFirstValue("description");
        String category = form.getFirstValue("category");
        boolean withdrawn = Boolean.valueOf(form.getFirstValue("withdrawn"));
        String tagsss = form.getFirstValue("tags");
        String tagss = tagsss.replaceAll("\\s","");
        List<String> tags  = new ArrayList<String>(Arrays.asList(tagss.split(",")));
        Set<String> set = new HashSet<>(tags);
        tags.clear();
        tags.addAll(set);

        //validate the values (in the general case)
        Product product = dataAccess.updateProduct(id, name, description, category, withdrawn, tags);

        return new JsonProductRepresentation(product);
    }

    @Override
    protected Representation patch(Representation entity) throws ResourceException {

        String idAttr = getAttribute("id");

        if (idAttr == null) {
            throw new ResourceException(Status.CLIENT_ERROR_BAD_REQUEST, "Missing product id");
        }

        Long id = null;
        try {
            id = Long.parseLong(idAttr);
        }
        catch(Exception e) {
            throw new ResourceException(Status.CLIENT_ERROR_BAD_REQUEST, "Invalid product id: " + idAttr);
        }

        String key = new String("");
        String value = new String("");

        //Create a new restlet form
        Form form = new Form(entity);
        //Read the parameters
        String name = form.getFirstValue("name"); if (name != null){ key = "name"; value = name;}
        String description = form.getFirstValue("description"); if (description != null){ key = "description"; value = description;}
        String category = form.getFirstValue("category"); if (category != null){ key = "category"; value = category;}
        String withdrawn = form.getFirstValue("withdrawn"); if (withdrawn != null){ key = "withdrawn"; value = withdrawn;}
        String tags = form.getFirstValue("tags"); if (tags != null){ key = "tags"; value = tags;}

        //validate the values (in the general case)
        Product product = dataAccess.partialUpdateProduct(id, key, value);

        return new JsonProductRepresentation(product);
    }

    @Override
    protected Representation delete() throws ResourceException {

      String idAttr = getAttribute("id");

      if (idAttr == null) {
          throw new ResourceException(Status.CLIENT_ERROR_BAD_REQUEST, "Missing product id");
      }

      Long id = null;
      try {
          id = Long.parseLong(idAttr);
      }
      catch(Exception e) {
          throw new ResourceException(Status.CLIENT_ERROR_BAD_REQUEST, "Invalid product id: " + idAttr);
      }

      Series headers = (Series) getRequestAttributes().get("org.restlet.http.headers");
      String auth = headers.getFirstValue("X-OBSERVATORY-AUTH");

      boolean admin = auth.contains("admin");

      Message message = dataAccess.deleteProduct(id, admin);

      return new JsonMessageRepresentation(message);
    }
}
