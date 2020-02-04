package gr.ntua.ece.softeng18b.api;

import gr.ntua.ece.softeng18b.conf.Configuration;
import gr.ntua.ece.softeng18b.data.DataAccess;
import gr.ntua.ece.softeng18b.data.model.Product;
import gr.ntua.ece.softeng18b.data.model.Message;
import gr.ntua.ece.softeng18b.data.model.Shop;
import org.restlet.data.Status;
import org.restlet.representation.Representation;
import org.restlet.resource.ResourceException;
import org.restlet.resource.ServerResource;
import org.restlet.data.Form;

import java.util.*;

public class ShopResource extends ServerResource {
    private final DataAccess dataAccess = Configuration.getInstance().getDataAccess();

    @Override
    protected Representation get() throws ResourceException {

        String idAttr = getAttribute("id");

        if (idAttr == null) {
            throw new ResourceException(Status.CLIENT_ERROR_BAD_REQUEST, "Missing shop id");
        }

        Long id = null;
        try {
            id = Long.parseLong(idAttr);
        }
        catch(Exception e) {
            throw new ResourceException(Status.CLIENT_ERROR_BAD_REQUEST, "Invalid shop id: " + idAttr);
        }

        Optional<Shop> optional = dataAccess.getShop(id);
        Shop shop = optional.orElseThrow(() -> new ResourceException(Status.CLIENT_ERROR_NOT_FOUND, "Shop not found - id: " + idAttr));

        return new JsonShopRepresentation(shop);
    }

    protected Representation put(Representation entity) throws ResourceException {

        String idAttr = getAttribute("id");

        if (idAttr == null) {
            throw new ResourceException(Status.CLIENT_ERROR_BAD_REQUEST, "Missing shop id");
        }

        Long id = null;
        try {
            id = Long.parseLong(idAttr);
        }
        catch(Exception e) {
            throw new ResourceException(Status.CLIENT_ERROR_BAD_REQUEST, "Invalid shop id: " + idAttr);
        }

        //Create a new restlet form
        Form form = new Form(entity);
        //Read the parameters
        String name = form.getFirstValue("name");
        String address = form.getFirstValue("address");
        double lng = Double.valueOf(form.getFirstValue("lng"));
        double lat = Double.valueOf(form.getFirstValue("lat"));
        boolean withdrawn = Boolean.valueOf(form.getFirstValue("withdrawn"));
        String tagsss = form.getFirstValue("tags");
        String tagss = tagsss.replaceAll("\\s","");
        List<String> tags  = new ArrayList<String>(Arrays.asList(tagss.split(",")));
        Set<String> set = new HashSet<>(tags);
        tags.clear();
        tags.addAll(set);

        //validate the values (in the general case)
        Shop shop = dataAccess.updateShop(id, name, address, lng, lat, withdrawn, tags);

        return new JsonShopRepresentation(shop);
    }

    @Override
    protected Representation patch(Representation entity) throws ResourceException {

        String idAttr = getAttribute("id");

        if (idAttr == null) {
            throw new ResourceException(Status.CLIENT_ERROR_BAD_REQUEST, "Missing shop id");
        }

        Long id = null;
        try {
            id = Long.parseLong(idAttr);
        }
        catch(Exception e) {
            throw new ResourceException(Status.CLIENT_ERROR_BAD_REQUEST, "Invalid shop id: " + idAttr);
        }

        String key = new String("");
        String value = new String("");

        //Create a new restlet form
        Form form = new Form(entity);
        //Read the parameters
        String name = form.getFirstValue("name"); if (name != null){ key = "name"; value = name;}
        String address = form.getFirstValue("address"); if (address != null){ key = "address"; value = address;}
        String lng = form.getFirstValue("lng"); if (lng != null){ key = "lng"; value = lng;}
        String lat = form.getFirstValue("lat"); if (lat != null) { key = "lat"; value = lat;}
        String withdrawn = form.getFirstValue("withdrawn"); if (withdrawn != null){ key = "withdrawn"; value = withdrawn;}
        String tags = form.getFirstValue("tags"); if (tags != null){ key = "tags"; value = tags;}

        //validate the values (in the general case)
        Shop shop = dataAccess.partialUpdateShop(id, key, value);

        return new JsonShopRepresentation(shop);
    }

    @Override
    protected Representation delete() throws ResourceException {

        String idAttr = getAttribute("id");

        if (idAttr == null) {
            throw new ResourceException(Status.CLIENT_ERROR_BAD_REQUEST, "Missing shop id");
        }

        Long id = null;
        try {
            id = Long.parseLong(idAttr);
        }
        catch(Exception e) {
            throw new ResourceException(Status.CLIENT_ERROR_BAD_REQUEST, "Invalid shop id: " + idAttr);
        }

        boolean admin = true;

        Message message = dataAccess.deleteShop(id, admin);

        return new JsonMessageRepresentation(message);
    }
}