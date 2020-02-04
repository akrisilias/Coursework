package gr.ntua.ece.softeng18b.data;

import gr.ntua.ece.softeng18b.data.model.Price;
import gr.ntua.ece.softeng18b.data.model.Product;

import org.springframework.jdbc.core.RowMapper;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Date;
import java.util.List;

public class PriceRowMapper implements RowMapper {
    @Override
    public Price mapRow(ResultSet rs, int rowNum) throws SQLException {

        long id            = rs.getLong("id");
        double value        = rs.getDouble("value");
        Date date       = rs.getDate("date");
        long productId      = rs.getLong("productId");
        long shopId         = rs.getLong("shopId");
        String productName  = rs.getString("productName");
        String productTagss    = rs.getString("productTags");
        List<String> productTags  = new ArrayList<String>();
        if (productTagss != null)
            productTags  = new ArrayList<String>(Arrays.asList(productTagss.split(",")));
        String shopName     = rs.getString("shopName");
        String shopTagss    = rs.getString("shopTags");
        List<String> shopTags  = new ArrayList<String>();
        if (shopTagss != null)
            shopTags  = new ArrayList<String>(Arrays.asList(shopTagss.split(",")));
        String shopAddress  = rs.getString("shopAddress");
        int shopDist    = rs.getInt("dist");



        return new Price(id, value, date, productId, shopId, productName, productTags, shopName, shopTags, shopAddress, shopDist);
    }
}
