package gr.ntua.ece.softeng18b.client.rest

import gr.ntua.ece.softeng18b.client.model.Product
import gr.ntua.ece.softeng18b.client.model.Shop
import org.apache.http.client.fluent.Executor
import org.apache.http.client.fluent.Form
import org.apache.http.client.fluent.Request
import java.nio.charset.Charset

import java.text.SimpleDateFormat

class LowLevelAPI {

    static final def SORT_OPTIONS = [
        "id|ASC",
        "id|DESC",
        "name|ASC",
        "name|DESC"
    ]

    static final def STATUSES = [
        "ACTIVE",
        "WITHDRAWN",
        "ALL"
    ]

    static final String DEFAULT_SORT   = SORT_OPTIONS[0]
    static final String DEFAULT_STATUS = STATUSES[0]

    static final String BASE_PATH = "/observatory/api"
    static final String HEADER    = "X-OBSERVATORY-AUTH"

    static final String IGNORE_SSL_ERRORS_SYSTEM_PROPERTY = "IGNORE_SSL_ERRORS"

    static final SimpleDateFormat FORMAT = new SimpleDateFormat("YYYY-MM-DD")

    private final String host
    private final int port
    private final boolean secure
    private final ClientFactory clientFactory

    LowLevelAPI(String host, int port, boolean secure) {
        this.host   = host
        this.port   = port
        this.secure = secure
        this.clientFactory = determineClientFactory()
    }

    private String createUrl(String endPoint, RestCallFormat format, Map params = null ) {
        Map queryParams
        if (format == RestCallFormat.JSON) {
            queryParams = [:]
        }
        else {
            queryParams = [format: format.getName()]
        }
        if (params) queryParams.putAll(params)
        String queryString
        if  (queryParams) {
            queryString = "?" + encode(params)
        }
        else {
            queryString = ""
        }
        String url = "${secure ? 'https' : 'http'}://$host:$port$BASE_PATH/$endPoint$queryString"
        //println "Fetching $url"
        return url
    }

    protected RestCallResult execute(Request req, RestCallFormat format) {
        req = req.addHeader('content-type', 'application/x-www-form-urlencoded;charset=UTF-8')
        Executor.newInstance(clientFactory.newClient()).execute(req).handleResponse(new RestResponseHandler(format))
    }

    RestCallResult login(String username, String password, RestCallFormat format) {

        return execute(
            Request.Post(createUrl("login", format)).
            bodyForm(
                Form.form().
                    add("username", username).
                    add("password", password).
                build()
            ),
            format
        )
    }

    RestCallResult logout(String token, RestCallFormat format) {

        if (!token) throw new RuntimeException("Empty token")

        return execute(
            Request.Post(createUrl("logout", format)).addHeader(HEADER, token),
            format
        )
    }

    RestCallResult getProduct(String token, String id, RestCallFormat format) {

        def req = Request.Get(createUrl("products/$id", format))

        if (token) req.addHeader(HEADER, token)

        return execute(req, format)
    }

    RestCallResult getProducts(String token, int start, int count, String status, String sort, RestCallFormat format) {

        def req = Request.Get(createUrl("products", format, [
            start : start,
            count : count,
            status: status,
            sort  : sort
        ]))

        if (token) req.addHeader(HEADER, token)

        return execute(req, format)
    }

    RestCallResult postProduct(String token, Product product, RestCallFormat format) {

        if (!token) throw new RuntimeException("Empty token")

        Form form = Form.form()
        addToForm(form, product)

        return execute(
            Request.Post(createUrl("products", format)).bodyForm(form.build(), Charset.defaultCharset()).addHeader(HEADER, token),
            format
        )
    }

    RestCallResult putProduct(String token, String id, Product product, RestCallFormat format){

        if (!token) throw new RuntimeException("Empty token")

        Form form = Form.form()
        addToForm(form, product)

        return execute(
            Request.Put(createUrl("products/$id", format)).bodyForm(form.build()).addHeader(HEADER, token),
            format
        )
    }

    RestCallResult patchProduct(String token, String id, String field, def value, RestCallFormat format) {

        if (!token) throw new RuntimeException("Empty token")

        Form form = Form.form()
        addFieldToForm(form, field, value)

        return execute(
            Request.Patch(createUrl("products/$id", format)).bodyForm(form.build()).addHeader(HEADER, token),
            format
        )
    }

    RestCallResult deleteProduct(String token, String id, RestCallFormat format) {

        if (!token) throw new RuntimeException("Empty token")

        return execute(
            Request.Delete(createUrl("products/$id", format)).addHeader(HEADER, token),
            format
        )
    }

    RestCallResult getShop(String token, String id, RestCallFormat format) {

        def req = Request.Get(createUrl("shops/$id", format))

        if (token) req.addHeader(HEADER, token)

        return execute(req, format)
    }

    RestCallResult getShops(String token, int start, int count, String status, String sort, RestCallFormat format) {

        def req = Request.Get(createUrl("shops", format, [
            start : start,
            count : count,
            status: status,
            sort  : sort
        ]))

        if (token) req.addHeader(HEADER, token)

        return execute(req, format)
    }

    RestCallResult postShop(String token, Shop shop, RestCallFormat format) {

        if (!token) throw new RuntimeException("Empty token")

        Form form = Form.form()
        addToForm(form, shop)

        return execute(
            Request.Post(createUrl("shops", format)).bodyForm(form.build(), Charset.defaultCharset()).addHeader(HEADER, token),
            format
        )
    }

    RestCallResult putShop(String token, String id, Shop shop, RestCallFormat format) {

        if (!token) throw new RuntimeException("Empty token")

        Form form = Form.form()
        addToForm(form, shop)

        return execute(
            Request.Put(createUrl("shops/$id", format)).bodyForm(form.build()).addHeader(HEADER, token),
            format
        )
    }

    RestCallResult patchShop(String token, String id, String field, def value, RestCallFormat format) {

        if (!token) throw new RuntimeException("Empty token")

        Form form = Form.form()
        addFieldToForm(form, field, value)

        return execute(
            Request.Patch(createUrl("shops/$id", format)).bodyForm(form.build()).addHeader(HEADER, token),
            format
        )
    }

    RestCallResult deleteShop(String token, String id, RestCallFormat format) {

        if (!token) throw new RuntimeException("Empty token")

        return execute(
            Request.Delete(createUrl("shops/$id", format)).addHeader(HEADER, token),
            format
        )
    }

    RestCallResult postPrice(
        String token,
        double price,
        String dateFrom,
        String dateTo,
        String productId,
        String shopId,
        RestCallFormat format) {

        if (!token) throw new RuntimeException("Empty token")

        Form form = Form.form()
        addFieldToForm(form, "price", price)
        addFieldToForm(form, "dateFrom", dateFrom)
        addFieldToForm(form, "dateTo", dateTo)
        addFieldToForm(form, "productId", productId)
        addFieldToForm(form, "shopId", shopId)

        return execute(
            Request.Post(createUrl("prices", format)).bodyForm(form.build(), Charset.defaultCharset()).addHeader(HEADER, token),
            format
        )

    }

    RestCallResult getPrices(
        String token,
        int start,
        int count,
        Integer geoDist,
        Double geoLng,
        Double geoLat,
        String dateFrom,
        String dateTo,
        List<String> shopIds,
        List<String> productIds,
        List<String> tags,
        List<String> sort,
        RestCallFormat format) {

        if (!token) throw new RuntimeException("Empty token")

        return execute(
            Request.Get(createUrl("prices", format, [
                start: start,
                count: count,
                geoDist: geoDist,
                geoLng: geoLng,
                geoLat: geoLat,
                dateFrom: dateFrom,
                dateTo: dateTo,
                shops: shopIds,
                products: productIds,
                tags: tags,
                sort: sort
            ])).addHeader(HEADER, token),
            format
        )
    }

    private static void addToForm(Form form, def item) {
        item.class.declaredFields.each {
            if (!it.synthetic) {
                addFieldToForm(form, it.name, item[(it.name)])
            }
        }
    }

    private static void addFieldToForm(Form form, String field, def value) {
        if (value != null) {
            if (value instanceof List) {
                value.each { v ->
                    if (v) form.add(field, v.toString())
                }
            }
            else {
                form.add(field, value.toString())
            }
        }
    }

    private static String encodeForm(Form form) {
        def list = form.build()
        def p = list.collect {
            return it.getName() + "=" + URLEncoder.encode(it.getValue(), 'UTF-8')
        }
        return p.join("&")
    }

    static String encode(Map params) {
        Form form = Form.form()
        params.each { k, v ->
            addFieldToForm(form, k, v)
        }
        return encodeForm(form)
    }

    static String encode(Object o) {
        Form form = Form.form()
        addToForm(form, o)
        return encodeForm(form)
    }

    private static ClientFactory determineClientFactory() {
        String prop = System.getProperty(IGNORE_SSL_ERRORS_SYSTEM_PROPERTY, "false")
        return Boolean.parseBoolean(prop) ? new SSLErrorTolerantClientFactory() : new DefaultClientFactory()
    }
}

