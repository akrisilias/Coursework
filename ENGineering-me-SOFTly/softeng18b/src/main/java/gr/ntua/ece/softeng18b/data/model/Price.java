package gr.ntua.ece.softeng18b.data.model;

import java.text.SimpleDateFormat;
import java.time.LocalDate;
import java.util.Date;
import java.util.Objects;
import java.util.List;

public class Price {
    private final long id;
    private final double price;
    private final String date;
    private final long productId;
    private final long shopId;
    private final String productName;
    private final List<String> productTags;
    private final String shopName;
    private final List<String> shopTags;
    private final String shopAddress;
    private final int shopDist;

    public Price(long id, double price, Date date, long productId, long shopId, String productName, List<String> productTags,
                 String shopName, List<String> shopTags, String shopAddress, int shopDist) {
        this.id = id;
        this.price = price;
        this.date = new SimpleDateFormat("yyyy-MM-dd").format(date);
        this.productId = productId;
        this.shopId = shopId;
        this.productName = productName;
        this.productTags = productTags;
        this.shopName = shopName;
        this.shopTags = shopTags;
        this.shopAddress = shopAddress;
        this.shopDist = shopDist;



    }

        public long getId() {return id;}
        public double getPrice() {return price;}
        public String getDate() {return date;}
        public long getProductId() {return productId;}
        public long getShopId() {return shopId;}
        public String getProductName() {return productName;}
        public List<String> getProductTags() {return productTags;}
        public String getShopName() {return shopName;}
        public List<String> getShopTags() {return shopTags;}
        public String getShopAddress() {return shopAddress;}
        public int getShopDist() {return shopDist;}

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        Price price = (Price) o;
        return id == price.id;
    }

    @Override
    public int hashCode() {
        return Objects.hash(id);
    }
}
