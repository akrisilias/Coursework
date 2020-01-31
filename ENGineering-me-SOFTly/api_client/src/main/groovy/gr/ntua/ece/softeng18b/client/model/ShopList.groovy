package gr.ntua.ece.softeng18b.client.model

import groovy.transform.Canonical 

@Canonical class ShopList extends Paging {
    
    List<Shop> shops
    
}

