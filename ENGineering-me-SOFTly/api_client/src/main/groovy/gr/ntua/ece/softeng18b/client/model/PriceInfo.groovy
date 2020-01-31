package gr.ntua.ece.softeng18b.client.model

import groovy.transform.Canonical

@Canonical class PriceInfo {

    Double price
    String date
    String productName
    String productId
    List<String> productTags
    String shopId
    String shopName
    List<String> shopTags
    String shopAddress
    Integer shopDist

}
