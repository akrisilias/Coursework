package gr.ntua.ece.softeng18b.api;

import gr.ntua.ece.softeng18b.conf.Configuration;
import gr.ntua.ece.softeng18b.data.DataAccess;
import gr.ntua.ece.softeng18b.data.Limits;
import gr.ntua.ece.softeng18b.data.model.Product;
import org.restlet.data.Form;
import org.restlet.representation.Representation;
import org.restlet.resource.ResourceException;
import org.restlet.resource.ServerResource;
import gr.ntua.ece.softeng18b.data.model.Message;

import org.restlet.util.Series;

import java.util.HashMap;
import java.util.List;
import java.util.Arrays;
import java.util.Map;
import java.util.ArrayList;
import java.util.Set;
import java.util.HashSet;

import javax.mail.Header;

public class ProductsResource extends ServerResource {

    private final DataAccess dataAccess = Configuration.getInstance().getDataAccess();

    @Override
    protected Representation get() throws ResourceException {

        int start = 0;
        int count = 20;

        String start_str = getQueryValue("start");
        if (start_str != null) start = Integer.parseInt(start_str);

        String count_str = getQueryValue("count");
        if (count_str != null) count = Integer.parseInt(count_str);

        long total = dataAccess.numberOfRows("product");

        String status = getQueryValue("status"); //status = "";
        if (status == null) status = "where withdrawn = 0";
        else if (status.equals("ALL")) status = "";
        else if (status.equals("WITHDRAWN")) status = "where withdrawn = 1";
        else status = "where withdrawn = 0";

        String sort = getQueryValue("sort");
        if (sort == null) sort = "order by id desc";
        if (sort.equals("id|ASC")) sort = "order by id";
        else if (sort.equals("name|ASC")) sort = "order by name";
        else if (sort.equals("name|DESC")) sort = "order by name desc";
        else sort = "order by id desc";

        List<Product> products = dataAccess.getProducts(new Limits(start, count), status, sort);

        Map<String, Object> map = new HashMap<>();
        map.put("start", start);
        map.put("count", count);
        map.put("total", total);
        map.put("products", products);

        return new JsonMapRepresentation(map);
    }

    @Override
    protected Representation post(Representation entity) throws ResourceException {

        //Create a new restlet form
        Form form = new Form(entity);
        // get authentication token
        Series headers = (Series) getRequestAttributes().get("org.restlet.http.headers");
        String auth = headers.getFirstValue("X-OBSERVATORY-AUTH");
        //Read the parameters
        String name = form.getFirstValue("name");
        String description = form.getFirstValue("description");
        String category = form.getFirstValue("category");
        boolean withdrawn = Boolean.valueOf(form.getFirstValue("withdrawn"));
        String[] tags2 = form.getValuesArray("tags");
        List<String> tags1 = Arrays.asList(tags2);
        List<String> tags;
        if (tags1.size() == 1) {
          String tagsss = tags1.get(0);
          String tagss = tagsss.replaceAll("\\s,\\s",",");
          tags  = new ArrayList<String>(Arrays.asList(tagss.split(",")));
          Set<String> set = new HashSet<>(tags);
          tags.clear();
          tags.addAll(set);
        }
        else {
          tags = tags1;
        }

        //validate the values (in the general case)
        //...
        Product product = dataAccess.addProduct(name, description, category, withdrawn, tags);
        return new JsonProductRepresentation(product);

    }
}
