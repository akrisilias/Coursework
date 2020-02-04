package gr.ntua.ece.softeng18b.data;


import gr.ntua.ece.softeng18b.data.model.*;
import org.apache.commons.dbcp2.BasicDataSource;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.jdbc.core.PreparedStatementCreator;
import org.springframework.jdbc.core.RowMapper;
import org.springframework.jdbc.support.GeneratedKeyHolder;

import javax.sql.DataSource;
import java.sql.*;
import java.sql.Date;
import java.text.SimpleDateFormat;
import java.time.LocalDate;
import java.util.*;

import org.springframework.dao.EmptyResultDataAccessException;
import java.security.SecureRandom;

import org.restlet.resource.ResourceException;
public class DataAccess {


    private static final Object[] EMPTY_ARGS = new Object[0];

    private static final int MAX_TOTAL_CONNECTIONS = 18;
    private static final int MAX_IDLE_CONNECTIONS = 6;

    private DataSource dataSource;
    private JdbcTemplate jdbcTemplate;

    public void setup(String driverClass, String url, String user, String pass) throws SQLException {

        //initialize the data source
        BasicDataSource bds = new BasicDataSource();
        bds.setDriverClassName(driverClass);
        bds.setUrl(url);
        bds.setMaxTotal(MAX_TOTAL_CONNECTIONS);
        bds.setMaxIdle(MAX_IDLE_CONNECTIONS);
        bds.setUsername(user);
        bds.setPassword(pass);
        bds.setValidationQuery("SELECT 1");
        bds.setTestOnBorrow(true);
        bds.setDefaultAutoCommit(true);

        //check that everything works OK
        bds.getConnection().close();

        //initialize the jdbc template utilitiy
        jdbcTemplate = new JdbcTemplate(bds);
    }

    public List<Product> getProducts(Limits limits, String status, String sort) {
        String sel_query = "select product.*, group_concat(tag) as tags " +
                            "from product left join product_tags " +
                            "on id = pid " + status +
                            " group by id " + sort +
                            " limit " + limits.getCount() + " offset " + limits.getStart()
                            ;
        return jdbcTemplate.query(sel_query, EMPTY_ARGS, new ProductRowMapper());
    }

    public long numberOfRows(String table) {
      return jdbcTemplate.queryForObject("select count(id) from " + table, Long.class);
    }

    public List<Shop> getShops(Limits limits, String status, String sort) {
        String sel_query = "select shop.*, group_concat(tag) as tags " +
                "from shop left join shop_tags " +
                "on id = sid " + status +
                " group by id " + sort +
                " limit " + limits.getCount() + " offset " + limits.getStart()
                ;
        return jdbcTemplate.query(sel_query, EMPTY_ARGS, new ShopRowMapper());
    }

    public List<Price> getPrices(Limits limits, List<String> sort, double geoLng, double geoLat, int geoDist, LocalDate dateFrom,LocalDate dateTo,
                                 List<String> shops, List<String> products, List<String> tags) {
                String tags_str = "";
                if (!(tags.isEmpty())) {
                    for (int i = 0; i < tags.size(); i++) {
                        if (i == 0) {
                            tags_str = tags_str + "'" + (String) tags.get(i) + "'";
                        } else {
                            tags_str = tags_str + ",'" + (String) tags.get(i) + "'";
                        }
                    }
                }
                else{
                    try {

                        String get_all_tags = "SELECT product_tags.tag FROM product LEFT JOIN product_tags ON product.id = product_tags.pid" +
                                " UNION " +
                                "SELECT shop_tags.tag FROM shop LEFT JOIN shop_tags ON shop.id = shop_tags.sid";

                        List<Map<String,Object>> tags_temp = jdbcTemplate.queryForList(get_all_tags);
                        List<String> temp = new ArrayList<String>();
                        for(Map i:tags_temp){
                            temp.addAll(i.values());
                        }

                        for (int j = 0; j < temp.size(); j++) {
                            if (j == 0) {
                                tags_str = tags_str + "'" + (String) temp.get(j) + "'";
                            } else {
                                tags_str = tags_str + ",'" + (String) temp.get(j) + "'";
                            }
                        }

                    }
                    catch (EmptyResultDataAccessException e) {
                        tags_str = "0";
                    }
                }

//                if (products == null) {
//                    String sel_query2 = "select product.id from product group  by id";
//                    products = jdbcTemplate.query(sel_query2, EMPTY_ARGS, new ProductRowMapper());
//                }
                String products_str = "";
                if (!(products.isEmpty())) {
                    for (int i = 0; i < products.size(); i++) {
                        if (i == 0) {
                            products_str = products_str + "'" + (String) products.get(i) + "'";
                        } else {
                            products_str = products_str + ",'" + (String) products.get(i) + "'";
                        }
                    }
                }
                else {
                    products_str = "0";
                    

//                    try {
//
//                        String get_products_id = "select product.id from product";
////                        List<Map<String,Object>> products_temp = jdbcTemplate.queryForList(get_products_id);
//                        List<String> temp = new ArrayList<String>();
//
//
//
//
//
//                        for (int j = 0; j < temp.size(); j++) {
//                            if (j == 0) {
//                                products_str = products_str + "'" + (String) temp.get(j) + "'";
//                            } else {
//                                products_str = products_str + ",'" + (String) temp.get(j) + "'";
//                            }
//                        }
//                    }
//                    catch (EmptyResultDataAccessException e) {
//                        products_str = "0";
//                    }
                }

                String shops_str = "";
                if (shops != null) {
                    for (int i = 0; i < shops.size(); i++) {
                        if (i == 0) {
                            shops_str = shops_str + "'" + (String) shops.get(i) + "'";
                        } else {
                            shops_str = shops_str + ",'" + (String) shops.get(i) + "'";
                        }
                    }
                }
                else {
                    shops_str = "0";
//                    try {
//                        String get_shops_id = "select shop.id from shop";
//                        shops_str = jdbcTemplate.queryForObject(get_shops_id,String.class);
//                    }
//                    catch (EmptyResultDataAccessException e) {
//                        shops_str = "0";
//                    }
                }

                String sort_str = "";
                String sort_temp;
                for (int i = 0;i < sort.size();i++) {
                    sort_temp = (String)sort.get(i);
                    if (sort_temp.equals("price")) {
                        sort_temp = "value";
                        sort.set(i,sort_temp);
                    }
                    if (i == 0) {
                        sort_str = sort_str + "" + (String)sort.get(i) + "";
                    }
                    else {
                        sort_str = sort_str + " " + (String)sort.get(i) + " ";
                    }
                }





                 String    sel_query = "select price.id as id,price.value AS value, product.id AS productId,product.name AS productName, group_concat(DISTINCT product_tags.tag) as productTags, " +
                            " shop.id AS shopId, shop.name AS shopName, shop.address AS shopAddress, group_concat(DISTINCT shop_tags.tag) AS shopTags, " +
                            " distanceOf(shop.lng,shop.lat," + geoLng + "," + geoLat + ") AS dist, price.date as date " +
                            " FROM price LEFT JOIN " +
                            "(product LEFT JOIN product_tags " +
                            "ON  product.id = product_tags.pid) " +
                            "ON price.productId = product.id " +
                            "LEFT JOIN " +
                            "(shop LEFT JOIN shop_tags " +
                            "ON shop.id = shop_tags.sid) " +
                            "ON price.shopId = shop.id " +
                            "WHERE distanceOf(shop.lng,shop.lat," + geoLng + "," + geoLat + ") < " + geoDist + " AND (product.id IN (" + products_str + ") ) AND " +
                            "(shop.id IN (" + shops_str + ") ) AND (product_tags.tag IN (" + tags_str + ") OR  shop_tags.tag IN (" + tags_str + ") ) AND " +
                            " price.date BETWEEN '" + dateFrom + "' AND '" + dateTo + "' " +
                            " GROUP BY  price.value, product.id, shop.id " +
                            " ORDER BY " + sort_str +
                            " limit " + limits.getCount() + " offset " + limits.getStart();

                return jdbcTemplate.query(sel_query, EMPTY_ARGS, new PriceRowMapper());



    }

    public Product addProduct(String name, String description, String category, boolean withdrawn, List<String> tags ) {
        //Create the new product record using a prepared statement
        PreparedStatementCreator psc = new PreparedStatementCreator() {
            @Override
            public PreparedStatement createPreparedStatement(Connection con) throws SQLException {
                PreparedStatement ps = con.prepareStatement(
                        "insert into product(name, description, category, withdrawn) values(?, ?, ?, ?)",
                        Statement.RETURN_GENERATED_KEYS
                );
                ps.setString(1, name);
                ps.setString(2, description);
                ps.setString(3, category);
                ps.setBoolean(4, withdrawn);
                return ps;
            }
        };
        GeneratedKeyHolder keyHolder = new GeneratedKeyHolder();
        int cnt = jdbcTemplate.update(psc, keyHolder);

        if (cnt == 1) {
            //New row has been added
            Product product = new Product(
                keyHolder.getKey().longValue(), //the newly created project id
                name,
                description,
                category,
                withdrawn,
                tags
            );
            long id = product.getId();
            for (int i=0; i<tags.size(); i++) {
              String ins_query = "insert into product_tags(pid, tag) " +
                                    "values (" + id + ", '" + tags.get(i) + "')";
              jdbcTemplate.update(ins_query);
            }
            return product;

        }
        else {
            throw new RuntimeException("Creation of Product failed");
        }
    }

    public Shop addShop(String name, String address, double lng, double lat, boolean withdrawn, List<String> tags ) {
        //Create the new shop record using a prepared statement
        PreparedStatementCreator psc = new PreparedStatementCreator() {
            @Override
            public PreparedStatement createPreparedStatement(Connection con) throws SQLException {
                PreparedStatement ps = con.prepareStatement(
                        "insert into shop(name, address, lng, lat, withdrawn) values(?, ?, ?, ?, ?)",
                        Statement.RETURN_GENERATED_KEYS
                );
                ps.setString(1, name);
                ps.setString(2, address);
                ps.setDouble(3, lng);
                ps.setDouble(4, lat);
                ps.setBoolean(5, withdrawn);
                return ps;
            }
        };
        GeneratedKeyHolder keyHolder = new GeneratedKeyHolder();
        int cnt = jdbcTemplate.update(psc, keyHolder);

        if (cnt == 1) {
            //New row has been added
            Shop shop = new Shop(
                    keyHolder.getKey().longValue(), //the newly created project id
                    name,
                    address,
                    lng,
                    lat,
                    withdrawn,
                    tags
            );
            long id = shop.getId();
            for (int i=0; i<tags.size(); i++) {
                String ins_query = "insert into shop_tags(sid, tag) " +
                        "values (" + id + ", '" + tags.get(i) + "')";
                jdbcTemplate.update(ins_query);
            }
            return shop;

        }
        else {
            throw new RuntimeException("Creation of Shop failed");
        }
    }

    public Price addPrice (double value, java.sql.Date dateFrom, Date dateTo, long productId, long shopId) {
        PreparedStatementCreator psc = new PreparedStatementCreator() {
            @Override
            public PreparedStatement createPreparedStatement(Connection con) throws SQLException {

                PreparedStatement ps = con.prepareStatement(
                        "insert into price(value, date,  productId, shopId) values(?, ?, ?, ?)",
                        Statement.RETURN_GENERATED_KEYS
                );
                ps.setDouble(1, value);
                ps.setDate(2, dateFrom);
                ps.setLong(3, productId);
                ps.setLong(4, shopId);
                return ps;
            }
        };
        String date_temp;
        date_temp = new SimpleDateFormat("yyyy-MM-dd").format(dateFrom);

        long id;

        try {
            String check_query = "select price.id from price where price.productId = " + productId + " AND price.shopId = " + shopId +
                    " AND price.date = '" + date_temp + "'";
            id = jdbcTemplate.queryForObject(check_query,long.class);
        }
        catch (EmptyResultDataAccessException e) {
            id = 0;
        }

        if (id == 0) {
            GeneratedKeyHolder keyHolder = new GeneratedKeyHolder();
            int cnt = jdbcTemplate.update(psc, keyHolder);

            if (cnt == 1) {
                //New row has been added
                //get shop name
                String shopName_added = "";
                try {
                    String shopName_query = "select shop.name from shop where shop.id = " + shopId + "";
                    shopName_added = jdbcTemplate.queryForObject(shopName_query,String.class);
                }
                catch (EmptyResultDataAccessException e) {
                    shopName_added = "";
                }

                //get product name
                String productName_added = "";
                try {
                    String productName_query = "select product.name from product where product.id = " + productId + "";
                    productName_added = jdbcTemplate.queryForObject(productName_query,String.class);
                }
                catch (EmptyResultDataAccessException e) {
                    shopName_added = "";
                }

                //get shopAddress
                String shopAddress_added = "";
                try {
                    String shopAddress_query = "select shop.address from shop where shop.id = " + shopId + "";
                    shopAddress_added = jdbcTemplate.queryForObject(shopAddress_query,String.class);
                }
                catch (EmptyResultDataAccessException e) {
                    shopAddress_added = "";
                }

                int shopDist_added = 0;

                List<String> productTags_added = new ArrayList<String>(Arrays.asList("0"));
                List<String> shopTags_added = new ArrayList<String>(Arrays.asList("0"));

                Price price = new Price(
                        keyHolder.getKey().longValue(), //the newly created project id
                        value,
                        dateFrom,
                        productId,
                        shopId,
                        productName_added,
                        productTags_added,
                        shopName_added,
                        shopTags_added,
                        shopAddress_added,
                        shopDist_added
                );


                return price;

            } else {
                throw new RuntimeException("Creation of Price failed");
            }
        }
        else {
            return null;
//            throw new RuntimeException("Creation of Price failed");
        }
    }

    public Product updateProduct(long id, String name, String description, String category, boolean withdrawn, List<String> tags ) {
        //Update the product record using a prepared statement
        PreparedStatementCreator psc = new PreparedStatementCreator() {
            @Override
            public PreparedStatement createPreparedStatement(Connection con) throws SQLException {
                PreparedStatement ps = con.prepareStatement(
                        "update product set name = ?, description = ?, category = ?, withdrawn = ? where id = ?",
                        Statement.RETURN_GENERATED_KEYS
                );
                ps.setString(1, name);
                ps.setString(2, description);
                ps.setString(3, category);
                ps.setBoolean(4, withdrawn);
                ps.setLong(5, id);
                return ps;
            }
        };
        GeneratedKeyHolder keyHolder = new GeneratedKeyHolder();
        int cnt = jdbcTemplate.update(psc, keyHolder);

        if (cnt == 1) {
            String del_query = "delete from product_tags where pid = " + id ;
            jdbcTemplate.update(del_query);
            for (int i=0; i<tags.size(); i++) {
              String ins_query = "insert into product_tags(pid, tag) " +
                                    "values (" + id + ", '" + tags.get(i) + "')";
              jdbcTemplate.update(ins_query);
            }
            //One row has been affected
            Optional<Product> optional = this.getProduct(id);
            Product product = optional.get();
            return product;
        }
        else {
            throw new RuntimeException("Update of Product failed");
        }
    }

    public Shop updateShop(long id, String name, String address, double lng, double lat, boolean withdrawn, List<String> tags) {
        //Update the shop record using a prepared statement
        PreparedStatementCreator psc = new PreparedStatementCreator() {
            @Override
            public PreparedStatement createPreparedStatement(Connection con) throws SQLException {
                PreparedStatement ps = con.prepareStatement(
                        "update shop set name = ?, address = ?, lng = ?, lat=?, withdrawn = ? where id = ?",
                        Statement.RETURN_GENERATED_KEYS
                );
                ps.setString(1, name);
                ps.setString(2, address);
                ps.setDouble(3, lng);
                ps.setDouble(4, lat);
                ps.setBoolean(5, withdrawn);
                ps.setLong(6, id);
                return ps;
            }
        };
        GeneratedKeyHolder keyHolder = new GeneratedKeyHolder();
        int cnt = jdbcTemplate.update(psc, keyHolder);

        if (cnt == 1) {
            String del_query = "delete from shop_tags where sid = " + id ;
            jdbcTemplate.update(del_query);
            for (int i=0; i<tags.size(); i++) {
                String ins_query = "insert into shop_tags(sid, tag) " +
                        "values (" + id + ", '" + tags.get(i) + "')";
                jdbcTemplate.update(ins_query);
            }
            //One row has been affected
            Optional<Shop> optional = this.getShop(id);
            Shop shop = optional.get();
            return shop;
        }
        else {
            throw new RuntimeException("Update of Shop failed");
        }
    }

    public Product partialUpdateProduct(long id, String key, String value) {
        //Parially update the product record using a prepared statement
      if (key.equals("tags")){
        String tagss = value.replaceAll("\\s","");
        List<String> tags  = new ArrayList<String>(Arrays.asList(tagss.split(",")));
        Set<String> set = new HashSet<>(tags);
        tags.clear();
        tags.addAll(set);
        String del_query = "delete from product_tags where pid = " + id ;
        jdbcTemplate.update(del_query);
        int nrows;
        for (int i=0; i<tags.size(); i++) {
          String ins_query = "insert into product_tags(pid, tag) " +
                                "values (" + id + ", '" + tags.get(i) + "')";
          nrows = jdbcTemplate.update(ins_query);
          if (nrows != 1) throw new RuntimeException("Update of Product failed");
        }
        Optional<Product> optional = this.getProduct(id);
        Product product = optional.get();
        return product;
      }
      else {
        PreparedStatementCreator psc = new PreparedStatementCreator() {
            @Override
            public PreparedStatement createPreparedStatement(Connection con) throws SQLException {
                PreparedStatement ps = con.prepareStatement(
                        "update product set " + key + " = ? where id = ?",
                        Statement.RETURN_GENERATED_KEYS
                );
                //ps.setString(1, key);
                if (key.equals("withdrawn"))
                {
                  boolean boolean_value = Boolean.valueOf(value);
                  ps.setBoolean(1, boolean_value);
                }
                else
                {
                  ps.setString(1, value);
                }
                ps.setLong(2, id);
                return ps;
            }
        };
        GeneratedKeyHolder keyHolder = new GeneratedKeyHolder();
        int cnt = jdbcTemplate.update(psc, keyHolder);

        if (cnt == 1) {
            //One row has been affected
            Optional<Product> optional = this.getProduct(id);
            Product product = optional.get();
            return product;
        }
        else {
            throw new RuntimeException("Update of Product failed");
        }
      }
    }

    public Shop partialUpdateShop(long id, String key, String value) {
        //partially update the shop record using a prepared statement
        if (key.equals("tags")){
            String tagss = value.replaceAll("\\s","");
            List<String> tags  = new ArrayList<String>(Arrays.asList(tagss.split(",")));
            Set<String> set = new HashSet<>(tags);
            tags.clear();
            tags.addAll(set);
            String del_query = "delete from shop_tags where sid = " + id ;
            jdbcTemplate.update(del_query);
            int nrows;
            for (int i=0; i<tags.size(); i++) {
                String ins_query = "insert into shop_tags(sid, tag) " +
                        "values (" + id + ", '" + tags.get(i) + "')";
                nrows = jdbcTemplate.update(ins_query);
                if (nrows != 1) throw new RuntimeException("Update of Shop failed");
            }
            Optional<Shop> optional = this.getShop(id);
            Shop shop = optional.get();
            return shop;
        }
        else {
            PreparedStatementCreator psc = new PreparedStatementCreator() {
                @Override
                public PreparedStatement createPreparedStatement(Connection con) throws SQLException {
                    PreparedStatement ps = con.prepareStatement(
                            "update shop set " + key + " = ? where id = ?",
                            Statement.RETURN_GENERATED_KEYS
                    );
                    //ps.setString(1, key);
                    if (key.equals("withdrawn"))
                    {
                        boolean boolean_value = Boolean.valueOf(value);
                        ps.setBoolean(1, boolean_value);
                    }
                    else
                    {
                        ps.setString(1, value);
                    }
                    ps.setLong(2, id);
                    return ps;
                }
            };
            GeneratedKeyHolder keyHolder = new GeneratedKeyHolder();
            int cnt = jdbcTemplate.update(psc, keyHolder);

            if (cnt == 1) {
                //One row has been affected
                Optional<Shop> optional = this.getShop(id);
                Shop shop = optional.get();
                return shop;
            }
            else {
                throw new RuntimeException("Update of Shop failed");
            }
        }

    }

    public Message deleteProduct(long id, boolean admin) {
        //Parially update the product record using a prepared statement
        PreparedStatementCreator psc = new PreparedStatementCreator() {
            @Override
            public PreparedStatement createPreparedStatement(Connection con) throws SQLException {
                PreparedStatement ps;
                if (admin == true)
                {
                  ps = con.prepareStatement(
                    "delete from product where id = ?",
                    Statement.RETURN_GENERATED_KEYS
                  );
                  ps.setLong(1, id);
                }
                else
                {
                  ps = con.prepareStatement(
                    "update product set withdrawn = true where id = ?",
                    Statement.RETURN_GENERATED_KEYS
                  );
                  ps.setLong(1, id);
                }
                return ps;
            }
        };
        GeneratedKeyHolder keyHolder = new GeneratedKeyHolder();
        int cnt = jdbcTemplate.update(psc, keyHolder);

        if (cnt == 1) {
            //One row has been affected
            Message message = new Message("OK");
            return message;
        }
        else {
            throw new RuntimeException("Update of Product failed");
        }
    }

    public Message deleteShop(long id, boolean admin) {
        //Parially update the shop record using a prepared statement
        PreparedStatementCreator psc = new PreparedStatementCreator() {
            @Override
            public PreparedStatement createPreparedStatement(Connection con) throws SQLException {
                PreparedStatement ps;
                if (admin == true)
                {
                    ps = con.prepareStatement(
                            "delete from shop where id = ?",
                            Statement.RETURN_GENERATED_KEYS
                    );
                    ps.setLong(1, id);
                }
                else
                {
                    ps = con.prepareStatement(
                            "update shop set withdrawn = true where id = ?",
                            Statement.RETURN_GENERATED_KEYS
                    );
                    ps.setLong(1, id);
                }
                return ps;
            }
        };
        GeneratedKeyHolder keyHolder = new GeneratedKeyHolder();
        int cnt = jdbcTemplate.update(psc, keyHolder);

        if (cnt == 1) {
            //One row has been affected
            Message message = new Message("OK");
            return message;
        }
        else {
            throw new RuntimeException("Delete of Shop failed");
        }
    }

    public Token postLogin(String username, String password) {
      boolean exists = true;
      try {
        String sql = "select userID from users where username = '" + username + "' and password = '" + password + "'";
        int userID = jdbcTemplate.queryForObject(sql, Integer.class);
      }
      catch (EmptyResultDataAccessException e) {
        exists = false;
      }
      if (exists == true)
      {
        SecureRandom random = new SecureRandom();
        long longToken = Math.abs( random.nextLong() );
        String random2 = Long.toString( longToken, 16 );
          String sql = "select role from users where username = '" + username + "' and password = '" + password + "'";
          String role = jdbcTemplate.queryForObject(sql, String.class);
          String valid_message = role + ":" + random2;
        Token valid = new Token(valid_message);
        return valid;
      }
      else
      {
        String invalid_message = "Invalid input";
        Token invalid = new Token(invalid_message);
        return invalid;
      }
    }

    public Message postLogout() {
      Message message = new Message("OK");
      return message;
    }

    public Optional<Product> getProduct(long id) {
        Long[] params = new Long[]{id};
        String sel_query = "select product.*, group_concat(tag) as tags " +
                           "from product left join product_tags " +
                           "on id = pid " +
                           "where id = ?";
        List<Product> products = jdbcTemplate.query(sel_query, params, new ProductRowMapper());
        if (products.size() == 1)  {
            return Optional.of(products.get(0));
        }
        else {
            return Optional.empty();
        }
    }

    public Optional<Shop> getShop(long id) {
        Long[] params = new Long[]{id};
        String sel_query = "select shop.*, group_concat(tag) as tags " +
                "from shop left join shop_tags " +
                "on id = sid " +
                "where id = ?";
        List<Shop> shops = jdbcTemplate.query(sel_query, params, new ShopRowMapper());
        if (shops.size() == 1)  {
            return Optional.of(shops.get(0));
        }
        else {
            return Optional.empty();
        }
    }

}
