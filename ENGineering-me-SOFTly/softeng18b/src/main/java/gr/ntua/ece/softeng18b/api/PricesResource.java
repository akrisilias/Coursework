package gr.ntua.ece.softeng18b.api;

import gr.ntua.ece.softeng18b.conf.Configuration;
import gr.ntua.ece.softeng18b.data.DataAccess;
import gr.ntua.ece.softeng18b.data.Limits;
import gr.ntua.ece.softeng18b.data.model.Price;
import gr.ntua.ece.softeng18b.data.model.Product;
import gr.ntua.ece.softeng18b.data.model.Shop;
import org.restlet.data.Form;
import org.restlet.representation.Representation;
import org.restlet.resource.ResourceException;
import org.restlet.resource.ServerResource;

import java.io.IOException;
import java.text.DateFormat;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.time.LocalDate;
import java.time.format.DateTimeFormatter;
import java.util.*;

public class PricesResource extends ServerResource {
    private final DataAccess dataAccess = Configuration.getInstance().getDataAccess();

    @Override
    protected Representation get() throws ResourceException {

        int start = 0;
        int count = 20;

        String start_str = getQueryValue("start");
        if (start_str != null) start = Integer.parseInt(start_str);

        String count_str = getQueryValue("count");
        if (count_str != null) count = Integer.parseInt(count_str);

        long total = dataAccess.numberOfRows("price");

        List<String> sort;
        String sorttt = getQueryValue("sort");
        if (sorttt == null) {
            sort  = new ArrayList<String>(Arrays.asList("value","asc"));
        }
        else {
            String sortt = sorttt.replaceAll("\\s", "");
            sort = new ArrayList<String>(Arrays.asList(sortt.split("\\|")));
//            Set<String> set = new HashSet<>(sort);
//            sort.clear();
//            sort.addAll(set);
        }

        List<String> productId = new ArrayList<String>();

        Form query2 = getQuery();
        String[] products = query2.getValuesArray("products");
        String products_str = getQueryValue("products");
        if (products_str == null) {
            productId = null;
        }
        else {
            int l2 = products.length;
            for (int i = 0; i < l2; i++) {
                productId.add(products[i]);
            }
        }

        List<String> shopId = new ArrayList<String>();

            Form query = getQuery();
            String[] shops = query.getValuesArray("shops");
            String shops_str = getQueryValue("shops");
            if (shops_str == null) {
                shopId = null;
            }
            else {
                int l = shops.length;
                for (int j = 0; j < l; j++) {
                    shopId.add(shops[j]);
                }
            }


//        String shopIddd = getQueryValue("shops");
//        List<String> shopId = new ArrayList<String>();
//        while (shopIddd != null) {
//            shopId.add(shopIddd);
//            shopIddd = getQueryValue("shops");
//        }

//        if (shopIddd == null) {
////            shopId = new ArrayList<String>(Arrays.asList("0"));
//            shopId = null;
////            Set<String> set3 = new HashSet<>(shopId);
////            shopId.clear();
////            shopId.addAll(set3);
//        }
//        else {
//            String shopIdd = shopIddd.replaceAll("\\s", "");
//            shopId = new ArrayList<String>(Arrays.asList(shopIddd.split(",")));
//            Set<String> set3 = new HashSet<>(shopId);
//            shopId.clear();
//            shopId.addAll(set3);
//        }
//       String shopIddd2 = getQueryValue("shops");
//        while (shopIddd2 != null) {
//           String shopIdd = shopIddd2.replaceAll("\\s", "");
//            List<String> shopId_temp = new ArrayList<String>(Arrays.asList(shopIdd.split(",")));
//            Set<String> set5 = new HashSet<>(shopId_temp);
//            shopId_temp.clear();
//            shopId_temp.addAll(set5);
//            for (int k = 0; k < shopId_temp.size(); k++) {
//                shopId.add((String)shopId_temp.get(k));
//            }
//            shopIddd2 = getQueryValue("shops");
//       }

        String tagsss = getQueryValue("tags");
        List<String> tags;
        if (tagsss == null) {
            tags = new ArrayList<String>(Arrays.asList());
//            tags = null;
        }
        else {
            String tagss = tagsss.replaceAll("\\s", "");
            tags = new ArrayList<String>(Arrays.asList(tagss.split(",")));
            Set<String> set4 = new HashSet<>(tags);
            tags.clear();
            tags.addAll(set4);
        }

        int geoDist = 0;
        double geoLng = 0.0;
        double geoLat = 0.0;
        String geoDist_str = getQueryValue("geoDist");
        String geoLng_str = getQueryValue("geoLng");
        String geoLat_str = getQueryValue("geoLat");
//        if ((geoDist_str == null && geoLat_str == null && geoLng_str == null ) || !(geoDist_str != null && geoLat_str != null && geoLng_str != null) ){
////            return new JsonMapRepresentation(null); // error message gia input
//            geoLng = 23.07;
//            geoLat = 38.02;
//            geoDist = Integer.MAX_VALUE;
//        }
//        if ((geoDist_str == null && geoLat_str == null && geoLng_str == null ) ){
//            return new JsonMapRepresentation(null); // error message gia input
//            geoLng = 23.07;
//            geoLat = 38.02;
//            geoDist = Integer.MAX_VALUE;
//        }
        if (geoDist_str != null) geoDist = Integer.parseInt(geoDist_str);
        if (geoLng_str != null) geoLng = Double.parseDouble(geoLng_str);
        if (geoLat_str != null) geoLat = Double.parseDouble(geoLat_str);

        LocalDate dateFrom = LocalDate.now();
        LocalDate dateTo = LocalDate.now() ;
//        Date dateFrom = new Date();
//        Date dateTo = new Date();
        String dateFrom_str = getQueryValue("dateFrom");
        String dateTo_str = getQueryValue("dateTo");

        if ((dateFrom_str == null && dateTo_str != null) || (dateFrom_str != null) && dateTo_str == null ) {
            return new JsonMapRepresentation(null); // error message
        }

        DateFormat format = new SimpleDateFormat("yyyy-MM-dd");

        if (dateFrom_str == null && dateTo_str == null) {
            dateFrom = LocalDate.now();
            dateTo = LocalDate.now();
//            SimpleDateFormat formatter = new SimpleDateFormat ("yyyy-MM-dd");
//            Date today = new Date();
//            try {
//                dateFrom = formatter.parse(formatter.format(today)); // mporei na einai lathos
//                dateTo = formatter.parse(formatter.format(today)); // mporei na einai lathos
//            } catch (ParseException e) {
//                e.printStackTrace();
//            }
        }

        else {

            dateFrom = LocalDate.parse(dateFrom_str);
            dateTo = LocalDate.parse(dateTo_str);

//                try {
//                    dateFrom = format.parse(dateFrom_str);
//                    dateTo = format.parse(dateTo_str);
//                } catch (java.text.ParseException e) {
//                    e.printStackTrace();
//                }


//            if (dateTo_str != null) {
//                try {
//                    dateTo = format.parse(dateTo_str);
//                } catch (java.text.ParseException e) {
//                    e.printStackTrace();
//                }
//            }

        }

        if (geoLng_str == null && geoLat_str == null) {
            geoLng = 23.7;
            geoLat = 38.02;
        }

        if (geoDist_str == null) {
          geoDist = Integer.MAX_VALUE;
        }

        List<Price> prices = dataAccess.getPrices(new Limits(start, count),  sort, geoLng, geoLat, geoDist, dateFrom, dateTo, shopId, productId,
                tags);

        total = prices.size();

        Map<String, Object> map = new HashMap<>();
        map.put("start", start);
        map.put("count", count);
        map.put("total", total);
        map.put("prices", prices);

        return new JsonMapRepresentation(map);
    }

    @Override
    protected Representation post(Representation entity) throws ResourceException {

        //Create a new restlet form
        List<Price> prices = new ArrayList<Price>(Arrays.asList()) ;
        LocalDate localdateFrom;
        LocalDate localdateTo;
        Form form = new Form(entity);
        //Read the parameters
        double value = Double.valueOf(form.getFirstValue("price"));

        String dateFrom_str = form.getFirstValue("dateFrom");
        Date dateFrom_temp = null;
        try {
            dateFrom_temp = new SimpleDateFormat("yyyy-MM-dd").parse(dateFrom_str);
        } catch (ParseException e) {
            e.printStackTrace();
        }
        java.sql.Date dateFrom = new java.sql.Date(dateFrom_temp.getTime());

        String dateTo_str = form.getFirstValue("dateTo");
        Date dateTo_temp = null;
        try {
            dateTo_temp = new SimpleDateFormat("yyyy-MM-dd").parse(dateTo_str);
        } catch (ParseException e) {
            e.printStackTrace();
        }
        java.sql.Date dateTo = new java.sql.Date(dateTo_temp.getTime());

        long productId = Long.valueOf(form.getFirstValue("productId"));
        long shopId = Long.valueOf(form.getFirstValue("shopId"));
        //validate the values (in the general case)
        //...
        localdateFrom = LocalDate.parse(dateFrom_str);
        localdateTo = LocalDate.parse(dateTo_str);
        Map<String, Object> map = new HashMap<>();
        localdateTo = localdateTo.plusDays(1);

        long total = 0;

        while (!(localdateFrom.equals(localdateTo))) {

            Price price = dataAccess.addPrice(value, dateFrom, dateTo, productId, shopId);

            if (price != null)
                prices.add(price);
            localdateFrom = localdateFrom.plusDays(1);
            dateFrom = java.sql.Date.valueOf(localdateFrom);
            total++;
        }


        map.put("start", 0);
        map.put("count", 20);
        map.put("total", total);
        map.put("prices", prices);


        if (map != null)
            return new JsonMapRepresentation(map);
        else
            return new JsonMapRepresentation(null);

//      return new JsonPriceRepresentation(price);
    }
}
