import { Component, OnInit } from '@angular/core';
import { DataService } from '../data.productsService';

@Component({
  selector: 'app-price',
  templateUrl: './price.component.html',
  styleUrls: ['./price.component.scss']
})
export class PriceComponent implements OnInit {

  prices: any;

  Price: string = '';
  DateFrom: string = '';
  DateTo: string = '';
  ProductId: string = '';
  ShopId: string = '';

  found: boolean = false;

  error: string = '';

  view: number = 20;

  constructor(private data: DataService) { }

  ngOnInit() {

  }

  postClick() {

    console.log("Price");
    console.log(this.Price);
    console.log("DateFrom");
    console.log(this.DateFrom);
    console.log("DateTo");
    console.log(this.DateTo);
    console.log("Product ID");
    console.log(this.ProductId);
    console.log("Shop ID");
    console.log(this.ShopId);

    if (this.Price == '' || this.DateFrom == '' || this.DateTo == '' || this.ProductId == '' || this.ShopId == ''
     || this.Price==null || this.DateFrom==null || this.DateTo==null || this.ProductId==null || this.ShopId==null) {
      this.error = 'Please enter all fields';
    }
    else {
      this.error = '';

      this.data.postPrices(this.Price, this.DateFrom, this.DateTo, this.ProductId, this.ShopId)
      .subscribe(data => {
        this.prices = data;
        console.log(this.prices);
        }
      );

      this.found = true;
    }


  }

}
