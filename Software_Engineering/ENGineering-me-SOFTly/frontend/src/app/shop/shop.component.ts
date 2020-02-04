import { Component, OnInit } from '@angular/core';
import { DataService } from '../data.productsService';
import { ActivatedRoute, Router } from '@angular/router';
import { ShopsComponent } from '../shops/shops.component';
import Map from 'ol/Map';
import Tile from 'ol/layer/Tile';
import OSM from 'ol/source/OSM';
import View from 'ol/View';
import { toLonLat } from 'ol/proj.js';
import { fromLonLat } from 'ol/proj.js';
import VectorSource from 'ol/source/Vector';
import {Icon, Style} from 'ol/style';
import Point from 'ol/geom/Point';
import Feature from 'ol/Feature';
import {Tile as TileLayer, Vector as VectorLayer} from 'ol/layer';

@Component({
  selector: 'app-shop',
  templateUrl: './shop.component.html',
  styleUrls: ['./shop.component.scss']
})
export class ShopComponent implements OnInit {

  shop: any;
  map;
  vectorSource;
  vectorLayer;

  id: string;

  Name: string;
  Address: string;
  Longtitude: string;
  Latitude: string;
  Withdrawn: string;
  Tags: string;

  constructor(private data: DataService, private route: ActivatedRoute, private router: Router) { }

  ngOnInit() {
    this.id = this.route.snapshot.paramMap.get('id');

    this.data.getShop(this.id)
      .subscribe(data => {
        this.shop = data;
        console.log(this.shop);
        this.Name = this.shop.name;
        this.Address = this.shop.address;
        this.Longtitude = this.shop.lng;
        this.Latitude = this.shop.lat;
        this.Withdrawn = this.shop.withdrawn;
        this.Tags = this.shop.tags;
        }
      );
	  
	  this.initilizeMap();
  }

  putClick() {
    this.data.putShop(this.id, this.Name, this.Address, this.Longtitude, this.Latitude, this.Withdrawn, this.Tags)
      .subscribe( data => {
        console.log(data);
        this.router.navigate(['shops']);
        }
      );

  }
  
  initilizeMap() {
	  
	var coord = fromLonLat([23.71622, 37.97945]);
	
	this.vectorSource = new VectorSource();
    this.vectorLayer = new VectorLayer({
      source: this.vectorSource
    });

    this.map = new Map({
      target: 'map',
      layers: [
        new TileLayer({ source: new OSM() }),
        this.vectorLayer
      ],
      view: new View({
        center: coord,
        zoom: 10
      })
    });
	
	this.map.on('click', (evt) => {
		this.vectorSource.clear();
		var [coord1, coord2]=this.map.getEventCoordinate(evt.originalEvent);
		
		var lonlat = toLonLat([coord1, coord2]);
		var lon = lonlat[0];
		var lat = lonlat[1];
		this.Longtitude=lon;
		this.Latitude=lat;
		// …	
			console.log('clicked');
			
			
          let feature = new Feature(
            new Point([coord1, coord2])
          );

          let style= new Style({
			image: new Icon(({
			color: 'red',
			crossOrigin: 'anonymous',
			src: 'assets/img/dot.png',
			imgSize: [20, 20]
			}))
		})

          // let value = parseFloat(datum.value);
          feature.setStyle(style);
          this.vectorSource.addFeature(feature);
	
	});
  }
}
