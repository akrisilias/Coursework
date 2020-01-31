package gr.ntua.ece.softeng18b.client.model

import groovy.transform.Canonical 

@Canonical class ProductList extends Paging {	   
    
    List<Product> products
        
}

