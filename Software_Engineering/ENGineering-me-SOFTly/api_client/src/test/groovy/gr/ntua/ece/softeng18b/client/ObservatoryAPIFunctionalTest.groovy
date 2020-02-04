package gr.ntua.ece.softeng18b.client


import gr.ntua.ece.softeng18b.client.model.*
import gr.ntua.ece.softeng18b.client.rest.RestCallFormat
import groovy.json.JsonSlurper
import spock.lang.Shared
import spock.lang.Specification
import spock.lang.Stepwise
import spock.lang.Unroll

@Stepwise
class ObservatoryAPIFunctionalTest extends Specification {

    @Shared RestAPI api = null
    @Shared def testData = null
    @Shared def productIds = [] //product insertion index to product id
    @Shared def shopIds = [] //shop insertion index to product id

    def "initialize api client"(){
        when:
        String host     = System.getProperty("host")
        String port     = System.getProperty("port")
        String protocol = System.getProperty("protocol")
        api = new RestAPI(host, port as Integer, protocol == 'https')

        then:
        noExceptionThrown()
    }

    def "login"() {
        when:
        String username = System.getProperty("username")
        String password = System.getProperty("password")
        api.login(username, password, RestCallFormat.JSON)

        then:
        api.isLoggedIn()
    }

    def "setup test data"() {
        when:
        String pathToJson = System.getProperty("test.json")
        testData = new JsonSlurper().parse(new File(pathToJson), "utf-8")

        then:
        noExceptionThrown()
    }

    @Unroll
    def "post product #p"() {
        given:
        Product posted = new Product(
            name: p.name as String,
            description: p.description as String,
            category: p.category as String,
            tags: p.tags as List<String>
        )
        Product returned = api.postProduct(posted, RestCallFormat.JSON)
        productIds.push(returned.getId())

        expect:
        returned.name == posted.name &&
        returned.description == posted.description
        returned.category == posted.category &&
        returned.tags.toSorted() == posted.tags.toSorted() &&
        !returned.withdrawn

        where:
        p << testData.products

    }

    @Unroll
    def "fetch products query #q"() {
        given:
        ProductList list = api.getProducts(
            q.start as Integer,
            q.count as Integer,
            q.status as String,
            q.sort as String,
            RestCallFormat.JSON
        )

        expect:
        list.start == 0 &&
        list.total == q.results.size() &&
        list.products.collect { it.name } == q.results as List<String>

        where:
        q << testData.products_queries
    }

    @Unroll
    def "post shop #s"() {
        given:
        Shop posted = new Shop(
            name: s.name as String,
            address: s.address as String,
            lat: s.lat as double,
            lng: s.lng as double,
            tags: s.tags as List<String>
        )
        Shop returned = api.postShop(posted, RestCallFormat.JSON)
        shopIds.push(returned.getId())

        expect:
        returned.name == posted.name &&
        returned.address == posted.address &&
        returned.lat == posted.lat &&
        returned.lng == posted.lng &&
        returned.tags.toSorted() == posted.tags.toSorted() &&
        !returned.withdrawn

        where:
        s << testData.shops
    }

    @Unroll
    def "fetch shops query #q"() {
        given:
        ShopList list = api.getShops(
            q.start as Integer,
            q.count as Integer,
            q.status as String,
            q.sort as String,
            RestCallFormat.JSON
        )

        expect:
        list.start == 0 &&
        list.total == q.results.size() &&
        list.shops.collect { it.name } == q.results as List<String>

        where:
        q << testData.shops_queries
    }

    @Unroll
    def "post price #p"() {
        given:
        PriceInfoList list = api.postPrice(
            p.price as double,
            p.dateFrom as String,
            p.dateTo as String,
            productIds[p.productIndex as int] as String,
            shopIds[p.shopIndex as int] as String,
            RestCallFormat.JSON
        )
        expect:
        list.start == 0 &&
        list.total == Helper.durationInDays(p.dateFrom as String, p.dateTo as String) + 1 &&
        list.prices.every { PriceInfo pinfo ->
            pinfo.price == p.price &&
            pinfo.shopId == shopIds[p.shopIndex] &&
            pinfo.productId == productIds[p.productIndex]
        }

        where:
        p << testData.prices
    }

    @Unroll
    def "fetch prices query #q"() {
        given:
        PriceInfoList list = api.getPrices(
            q.start as int,
            q.count as int,
            q.geoDist as Integer,
            q.geoLng as Double,
            q.geoLat as Double,
            q.dateFrom as String,
            q.dateTo as String,
            Helper.elementsAt(shopIds, q.shopIndexes as List<Integer>),
            Helper.elementsAt(productIds, q.productIndexes as List<Integer>),
            q.tags as List<String>,
            q.sort as List<String>,
            RestCallFormat.JSON
        )

        expect:
        list.start == q.results.start &&
        list.count == q.results.count &&
        list.total == q.results.total &&
        (0..<list.total).every { i ->
            list.prices[i].price == q.results.prices[i].price &&
            list.prices[i].date  == q.results.prices[i].date &&
            list.prices[i].shopId == shopIds[q.results.prices[i].shopIndex] &&
            list.prices[i].productId == productIds[q.results.prices[i].productIndex]
        }

        where:
        q << testData.prices_queries
    }

    def "logout"() {
        when:
        api.logout(RestCallFormat.JSON)

        then:
        !api.isLoggedIn()
    }
}
