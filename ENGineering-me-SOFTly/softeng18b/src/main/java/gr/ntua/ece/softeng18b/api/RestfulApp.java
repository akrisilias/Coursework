package gr.ntua.ece.softeng18b.api;

import org.restlet.Application;
import org.restlet.Restlet;
import org.restlet.routing.Router;
import org.restlet.service.CorsService;

import java.util.HashSet;
import java.util.Set;
import java.util.Arrays;

public class RestfulApp extends Application {

    public RestfulApp() {
        CorsService corsService = new CorsService();
        corsService.setAllowedOrigins(new HashSet(Arrays.asList("*")));
        corsService.setAllowedCredentials(false);
        corsService.setAllowingAllRequestedHeaders(true);
        Set<String> allowHeaders = new HashSet<>();
        allowHeaders.add("X-OBSERVATORY-AUTH");
        allowHeaders.add("Content-Type");
        corsService.setAllowedHeaders(allowHeaders);
        corsService.setSkippingResourceForCorsOptions(true);
        getServices().add(corsService);
    }

    @Override
    public synchronized Restlet createInboundRoot() {

        Router router = new Router(getContext());

        //GET, POST
        router.attach("/products", ProductsResource.class);

        //GET, DELETE
        router.attach("/products/{id}", ProductResource.class);

        //GET, POST
        router.attach("/shops", ShopsResource.class);

        //GET, DELETE
        router.attach("/shops/{id}", ShopResource.class);

        //GET, POST
        router.attach("/prices", PricesResource.class);

        //POST
        router.attach("/login", LoginResource.class);

        //POST
        router.attach("/logout", LogoutResource.class);

        return router;
    }

}
