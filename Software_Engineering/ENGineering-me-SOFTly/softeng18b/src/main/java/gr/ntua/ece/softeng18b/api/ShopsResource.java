package gr.ntua.ece.softeng18b.api;

import gr.ntua.ece.softeng18b.conf.Configuration;
import gr.ntua.ece.softeng18b.data.DataAccess;
import gr.ntua.ece.softeng18b.data.Limits;
import gr.ntua.ece.softeng18b.data.model.Product;
import gr.ntua.ece.softeng18b.data.model.Shop;
import org.restlet.data.Form;
import org.restlet.representation.Representation;
import org.restlet.resource.ResourceException;
import org.restlet.resource.ServerResource;

import java.util.*;

public class ShopsResource extends ServerResource {

    private final DataAccess dataAccess = Configuration.getInstance().getDataAccess();

    @Override
    protected Representation get() throws ResourceException {

      int start = 0;
      int count = 20;

      String start_str = getQueryValue("start");
      if (start_str != null) start = Integer.parseInt(start_str);

      String count_str = getQueryValue("count");
      if (count_str != null) count = Integer.parseInt(count_str);

      long total = dataAccess.numberOfRows("shop");

      String status = getQueryValue("status"); //status = "";
      if (status == null) status = "where withdrawn = 0";
      else if (status.equals("ALL")) status = "";
      else if (status.equals("WITHDRAWN")) status = "where withdrawn = 1";
      else status = "where withdrawn = 0";

      String sort = getQueryValue("sort");
      if (sort == null) sort = "order by id desc";
      else if (sort.equals("id|ASC")) sort = "order by id";
      else if (sort.equals("name|ASC")) sort = "order by name";
      else if (sort.equals("name|DESC")) sort = "order by name desc";
      else sort = "order by id desc";

      List<Shop> shops = dataAccess.getShops(new Limits(start, count), status, sort);

      Map<String, Object> map = new HashMap<>();
      map.put("start", start);
      map.put("count", count);
      map.put("total", total);
      map.put("shops", shops);

      return new JsonMapRepresentation(map);
    }

    @Override
    protected Representation post(Representation entity) throws ResourceException {

        //Create a new restlet form
        Form form = new Form(entity);
        //Read the parameters
        String name = form.getFirstValue("name");
        String address = form.getFirstValue("address");
        double lng = Double.valueOf(form.getFirstValue("lng"));
        double lat = Double.valueOf(form.getFirstValue("lat"));
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

        Shop shop = dataAccess.addShop(name, address, lng, lat, withdrawn, tags);

        return new JsonShopRepresentation(shop);
    }
}
